// Package httpserver wires public OpenAI/Anthropic routes, admin, and middleware.
package httpserver

import (
	"context"
	"encoding/json"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/GreyGunG/grokbuild-proxy/internal/admin"
	"github.com/GreyGunG/grokbuild-proxy/internal/adminui"
	"github.com/GreyGunG/grokbuild-proxy/internal/anthropic"
	"github.com/GreyGunG/grokbuild-proxy/internal/config"
	"github.com/GreyGunG/grokbuild-proxy/internal/openai"
	"github.com/GreyGunG/grokbuild-proxy/internal/storage"
	"github.com/GreyGunG/grokbuild-proxy/internal/upstream"
)

// ModelLister lists upstream models for GET /v1/models.
type ModelLister interface {
	ListModels(ctx context.Context) (*upstream.ModelList, error)
}

// ClientStore looks up client keys.
type ClientStore interface {
	LookupClientByPlaintext(plaintext string) (storage.ClientKey, bool, error)
}

// Options configures the HTTP server.
type Options struct {
	Config   config.Config
	AdminKey string

	Store     ClientStore
	OpenAI    *openai.Handlers
	Anthropic *anthropic.Handlers
	Admin     *admin.Handlers
	ModelList ModelLister
	// Version shown on GET /
	Version string
	Logger  *slog.Logger
	Metrics *Metrics
	// ReadinessCacheTTL bounds how often unauthenticated /readyz may touch the
	// JSON credential store. Zero uses one second.
	ReadinessCacheTTL time.Duration
}

// clientAuth implements ClientAuthenticator.
type clientAuth struct {
	store ClientStore
}

func (c clientAuth) AuthenticateClient(plaintext string) (bool, error) {
	plaintext = strings.TrimSpace(plaintext)
	if plaintext == "" {
		return false, nil
	}
	if c.store == nil {
		return false, nil
	}
	_, ok, err := c.store.LookupClientByPlaintext(plaintext)
	return ok, err
}

// New builds the root http.Handler.
func New(opts Options) http.Handler {
	metrics := opts.Metrics
	if metrics == nil {
		metrics = &Metrics{}
	}
	mw := &Middleware{
		Clients: clientAuth{
			store: opts.Store,
		},
		AdminKey:       opts.AdminKey,
		MaxBody:        opts.Config.Limits.MaxBodyBytes,
		MaxConcurrent:  opts.Config.Limits.MaxConcurrent,
		RequestTimeout: opts.Config.RequestTimeout(),
		Logger:         opts.Logger,
		Metrics:        metrics,
	}

	mux := http.NewServeMux()
	readiness := &readinessCache{store: opts.Store, ttl: opts.ReadinessCacheTTL}

	// Unauthenticated health / probe.
	// Register without method-specific catch-all patterns that conflict in Go 1.22+.
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		handleHealthz(w, r)
	})
	// Alias /health for compatibility with BlacklistedAIProxy
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		handleHealthz(w, r)
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		handleReadyz(w, r, readiness)
	})
	mux.Handle("/metrics", metrics.Handler())
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		handleRoot(w, r, opts.Version)
	})

	// Client-authenticated API.
	api := http.NewServeMux()
	if opts.OpenAI != nil {
		api.HandleFunc("POST /v1/responses", opts.OpenAI.HandleResponses)
		api.HandleFunc("POST /v1/chat/completions", opts.OpenAI.HandleChatCompletions)
	}
	if opts.Anthropic != nil && opts.Config.Anthropic.Enabled {
		api.HandleFunc("POST /v1/messages", opts.Anthropic.HandleMessages)
		api.HandleFunc("POST /v1/messages/count_tokens", opts.Anthropic.HandleCountTokens)
	}
	api.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		handleModels(w, r, opts)
	})

	protected := Chain(api, mw.LimitConcurrency, mw.LimitBody, mw.RequireClient)
	mux.Handle("/v1/", protected)

	// Compatibility: BlacklistedAIProxy sends to /messages instead of /v1/messages.
	// HandleMessages does not depend on URL path, so this is safe.
	if opts.Anthropic != nil && opts.Config.Anthropic.Enabled {
		compatMsg := Chain(http.HandlerFunc(opts.Anthropic.HandleMessages), mw.LimitConcurrency, mw.LimitBody, mw.RequireClient)
		mux.Handle("POST /messages", compatMsg)
		compatTokens := Chain(http.HandlerFunc(opts.Anthropic.HandleCountTokens), mw.LimitConcurrency, mw.LimitBody, mw.RequireClient)
		mux.Handle("POST /messages/count_tokens", compatTokens)
	}

	// Admin Web UI (unauthenticated). Register before /admin/ API so exact
	// paths and /admin/ui/* assets are not swallowed by RequireAdmin.
	// GET patterns also match HEAD in Go 1.22+ ServeMux.
	mux.Handle("GET /admin", adminui.IndexHandler())
	mux.Handle("GET /admin/{$}", adminui.IndexHandler())
	mux.Handle("GET /admin/ui/", http.StripPrefix("/admin/ui/", adminui.AssetsHandler()))

	// Admin JSON API (own auth). Subtree /admin/* except more-specific
	// patterns registered above (exact /admin, /admin/, /admin/ui/).
	if opts.Admin != nil {
		opts.Admin.AdminKey = opts.AdminKey
		opts.Admin.Config = opts.Config
		if opts.Version != "" {
			opts.Admin.Version = opts.Version
		}
		mux.Handle("/admin/", opts.Admin.Handler())
	}

	return requireTrustedAdminHost(mw.Observe(mw.Timeout(mux)), opts.Config)
}

func requireTrustedAdminHost(next http.Handler, cfg config.Config) http.Handler {
	trusted := make(map[string]struct{}, len(cfg.AdminTrustedHosts)+4)
	trusted["localhost"] = struct{}{}
	for _, raw := range cfg.AdminTrustedHosts {
		if host, err := config.NormalizeTrustedHost(raw); err == nil {
			trusted[host] = struct{}{}
		}
	}
	if listenHost, _, err := net.SplitHostPort(strings.TrimSpace(cfg.Listen)); err == nil {
		listenHost = strings.Trim(listenHost, "[]")
		if listenHost != "" && listenHost != "0.0.0.0" && listenHost != "::" {
			if host, normalizeErr := config.NormalizeTrustedHost(listenHost); normalizeErr == nil {
				trusted[host] = struct{}{}
			}
		}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/admin" && !strings.HasPrefix(r.URL.Path, "/admin/") {
			next.ServeHTTP(w, r)
			return
		}
		host, err := config.NormalizeTrustedHost(r.Host)
		if err != nil {
			http.Error(w, "misdirected admin request", http.StatusMisdirectedRequest)
			return
		}
		if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
			next.ServeHTTP(w, r)
			return
		}
		if _, ok := trusted[host]; !ok {
			http.Error(w, "misdirected admin request", http.StatusMisdirectedRequest)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// NewServer builds an *http.Server with sensible timeouts.
func NewServer(addr string, handler http.Handler, requestTimeout time.Duration) *http.Server {
	if requestTimeout <= 0 {
		requestTimeout = 600 * time.Second
	}
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ErrorLog:          slog.NewLogLogger(slog.Default().Handler(), slog.LevelError),
		ReadHeaderTimeout: 10 * time.Second,
		// Do not set WriteTimeout: the request context provides a protocol-aware
		// bound and allows SSE handlers to emit an error before closing.
		IdleTimeout: 120 * time.Second,
	}
}

type credentialReader interface {
	ListCredentials() ([]storage.Credential, error)
}

type readinessSnapshot struct {
	status    int
	ready     bool
	reason    string
	available int
}

type readinessCache struct {
	mu      sync.Mutex
	store   ClientStore
	ttl     time.Duration
	expires time.Time
	value   readinessSnapshot
}

func (c *readinessCache) current() readinessSnapshot {
	if c == nil {
		return readinessSnapshot{status: http.StatusServiceUnavailable, reason: "credential store unavailable"}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	if !c.expires.IsZero() && now.Before(c.expires) {
		return c.value
	}
	ttl := c.ttl
	if ttl <= 0 {
		ttl = time.Second
	}
	c.value = computeReadiness(c.store, now)
	c.expires = now.Add(ttl)
	return c.value
}

func computeReadiness(store ClientStore, now time.Time) readinessSnapshot {
	reader, ok := store.(credentialReader)
	if !ok {
		return readinessSnapshot{status: http.StatusServiceUnavailable, reason: "credential store unavailable"}
	}
	creds, err := reader.ListCredentials()
	if err != nil {
		return readinessSnapshot{status: http.StatusServiceUnavailable, reason: "credential store unreadable"}
	}
	available := 0
	for _, credential := range creds {
		if !credential.Enabled {
			continue
		}
		if credential.CooldownUntil != nil && credential.CooldownUntil.After(now) {
			continue
		}
		if strings.TrimSpace(credential.AccessToken) == "" && strings.TrimSpace(credential.RefreshToken) == "" {
			continue
		}
		if !credential.ExpiresAt.IsZero() && !credential.ExpiresAt.After(now) &&
			strings.TrimSpace(credential.RefreshToken) == "" {
			continue
		}
		available++
	}
	if available == 0 {
		return readinessSnapshot{status: http.StatusServiceUnavailable, reason: "no usable credentials"}
	}
	return readinessSnapshot{status: http.StatusOK, ready: true, reason: "ready", available: available}
}

func handleReadyz(w http.ResponseWriter, r *http.Request, cache *readinessCache) {
	snapshot := cache.current()
	writeReadiness(w, r, snapshot.status, snapshot.ready, snapshot.reason, snapshot.available)
}

func writeReadiness(w http.ResponseWriter, r *http.Request, status int, ready bool, reason string, available int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if r.Method == http.MethodHead {
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"ready":                 ready,
		"reason":                reason,
		"available_credentials": available,
	})
}

func handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if r.Method != http.MethodHead {
		_, _ = w.Write([]byte("ok\n"))
	}
}

func handleRoot(w http.ResponseWriter, r *http.Request, version string) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if r.Method == http.MethodHead {
		return
	}
	msg := "grokbuild-proxy"
	if version != "" {
		msg += " " + version
	}
	_, _ = w.Write([]byte(msg + "\n"))
}

func handleModels(w http.ResponseWriter, r *http.Request, opts Options) {
	type modelRow struct {
		ID      string `json:"id"`
		Object  string `json:"object,omitempty"`
		Created int64  `json:"created,omitempty"`
		OwnedBy string `json:"owned_by,omitempty"`
		Name    string `json:"name,omitempty"`
	}

	rows := make([]modelRow, 0, 32)
	seen := map[string]struct{}{}
	add := func(id, owned string, created int64, name string) {
		id = strings.TrimSpace(id)
		if id == "" {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		seen[id] = struct{}{}
		if owned == "" {
			owned = "xai"
		}
		if name == "" {
			name = id
		}
		rows = append(rows, modelRow{
			ID:      id,
			Object:  "model",
			Created: created,
			OwnedBy: owned,
			Name:    name,
		})
	}

	var upstreamModels []upstream.Model
	if opts.ModelList != nil {
		if list, err := opts.ModelList.ListModels(r.Context()); err == nil && list != nil {
			upstreamModels = list.Data
			for _, m := range list.Data {
				add(m.ID, firstNonEmpty(m.OwnedBy, "xai"), m.Created, firstNonEmpty(m.Name, m.ID))
			}
		}
	}

	// Anthropic-facing aliases for Claude Code discovery.
	if opts.Config.Anthropic.Enabled {
		for _, e := range anthropic.AliasModels(upstreamModels, opts.Config.Anthropic) {
			add(e.ID, firstNonEmpty(e.OwnedBy, "anthropic"), e.Created, firstNonEmpty(e.Name, e.ID))
		}
	}

	// Always expose configured alias targets / short names as discovery aids.
	for alias, target := range opts.Config.Anthropic.ModelAliases {
		add(alias, "anthropic", 0, alias)
		add(target, "xai", 0, target)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   rows,
	})
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
