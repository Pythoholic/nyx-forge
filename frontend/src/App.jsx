import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle, Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import {
  Check,
  Copy,
  Download,
  ExternalLink,
  Eye,
  Heart,
  EyeOff,
  Image as ImageIcon,
  LoaderCircle,
  LayoutGrid,
  Layers,
  KeyRound,
  LogIn,
  Maximize2,
  Play,
  RotateCw,
  Save,
  Square,
  ShieldCheck,
  Smartphone,
  Sparkles,
  Star,
  UserPlus,
  WandSparkles,
  X,
} from "lucide-react";
import { setFavorite, getBackends, getBackendSettings, updateBackend, controlBackend, getVideoHistory, getVideoStatus, setVideoFavorite, getStorageSettings, setStorageSettings, beginAuthenticatorSetup, confirmAuthenticatorSetup, generateImage, getAuthSession, getHistory, getJob, getModels, getPromptModels, getSurprisePrompt, loginAccount, loginWithAuthenticator, logoutAccount, pixelUpscaleImage, rateImage, registerAccount, selectWorkspace, surpriseGenerateImage, testCloudCredential, transformImage, updateAccountSettings, upscaleImage } from "./api";
import { AppStatusBar, AppTopBar, NyxThemePicker, ROUTES, navigate, normalizeAppPath } from "./AppChrome";
import CustomSelect from "./CustomSelect";
import ErrorBanner from "./ErrorBanner";
import ExecutionJobs from "./ExecutionJobs";
import ForgeIMGPage from "./ForgeIMGPage";
import ForgeVIDPage from "./ForgeVIDPage";
import ForgeBATPage from "./ForgeBATPage.jsx";
import ForgeBATHome from "./ForgeBATHome.jsx";
import PhotoSwipeGallery from "./PhotoSwipeGallery";
import PromptSourceDialog from "./PromptSourceDialog";
import { displayQuality, displayStyle } from "./jobLabels";
import { NYX_THEMES, readNyxTheme } from "./nyxTheme";
import { useOperationsMonitor } from "./useOperationsMonitor";

const DashboardPage = lazy(() => import("./DashboardPage"));
const JobsPage = lazy(() => import("./JobsPage"));

const RATIOS = ["Portrait (832x1216)", "Landscape (1216x832)", "Square (1024x1024)"];
const RATINGS = ["Safe"];
const QUALITY_MODES = [["normal", "Normal"], ["high", "High"], ["super", "Super"], ["4k", "4K"], ["8k", "8K"], ["12k", "12K"]];
const QUALITY_SCALES = { normal: 1, high: 1.5, super: 2, "4k": 3, "8k": 8, "12k": 12 };

function qualityOutputSummary(mode, aspectRatio) {
  const match = String(aspectRatio).match(/(\d+)x(\d+)/);
  const scale = QUALITY_SCALES[mode] ?? 1;
  const size = match ? `${Math.round(Number(match[1]) * scale)}×${Math.round(Number(match[2]) * scale)}` : "model-native size";
  if (mode === "12k") return `${size} · Extreme staged output; expect a considerably longer run.`;
  if (mode === "8k") return `${size} · Large tiled output; takes considerably longer.`;
  if (mode === "4k") return `${size} · Detail pass followed by a 2× tiled upscale.`;
  if (mode === "super") return `${size} · Maximum standard refinement.`;
  if (mode === "high") return `${size} · Refined detail; recommended default.`;
  return `${size} · Native model resolution; fastest.`;
}
const FEEDBACK_REASONS = [
  { code: "prompt_match", positive: "Prompt match", negative: "Prompt mismatch" },
  { code: "eyes", positive: "Natural eyes", negative: "Eyes or gaze" },
  { code: "anatomy", positive: "Natural anatomy", negative: "Face or anatomy" },
  { code: "composition", positive: "Composition", negative: "Composition" },
  { code: "style", positive: "Style", negative: "Style mismatch" },
  { code: "detail", positive: "Fine detail", negative: "Detail or sharpness" },
  { code: "content_rating", positive: "Rating accuracy", negative: "Rating mismatch" },
];
const CLOUD_PROVIDERS = [
  { value: "deepseek", label: "DeepSeek", keyHint: "sk-…" },
  { value: "anthropic", label: "Anthropic", keyHint: "sk-ant-…" },
  { value: "venice", label: "Venice AI", keyHint: "Venice API key" },
];
const cloudProviderState = (initialValue) => Object.fromEntries(
  CLOUD_PROVIDERS.map(({ value }) => [value, initialValue]),
);
const LEGACY_HASH_ROUTES = Object.freeze({
  "#gallery": ROUTES.gallery,
  "#dashboard": ROUTES.analytics,
  "#jobs": ROUTES.deploy,
});
const DEPLOY_ROUTE_LABELS = Object.freeze({
  generate: "Image generation",
  surprise_generate: "Prompt and image",
  upscale: "Detail upscale",
  pixel_upscale: "Pixel upscale",
  img2img: "Image transform",
  character: "Same character",
});
const UNLOCKED_BOOT_KEY = "forge_unlocked_boot_id";
const WORKSPACES = [
  { id: "forgeai", label: "ForgeAI", code: "AI", category: "Generate", description: "Create new images from prompts, presets, and model-guided ideas.", route: ROUTES.create },
  { id: "forgeimg", label: "ForgeIMG", code: "IM", category: "Edit", description: "Transform existing images, preserve identities, and manage source assets.", route: ROUTES.img },
  { id: "forgevid", label: "ForgeVID", code: "VD", category: "Motion", description: "Generate motion from text or animate a gallery image.", route: ROUTES.vid },
  { id: "forgebat", label: "ForgeBAT", code: "BT", category: "Batch", description: "Define a visual search space and generate a varied batch from it.", route: ROUTES.bat },
];

function workspaceRoute(workspace) {
  return WORKSPACES.find((item) => item.id === workspace)?.route ?? ROUTES.create;
}

function initialRoutePath() {
  const legacy = LEGACY_HASH_ROUTES[window.location.hash];
  const path = legacy ?? normalizeAppPath();
  if (legacy || window.location.pathname === "/" || window.location.pathname.endsWith("/index.html")) {
    window.history.replaceState({}, "", path);
  }
  return path;
}

function pageForPath(path) {
  if (path.startsWith(ROUTES.bat)) return "bat";
  if (path.startsWith(ROUTES.vid)) return "vid";
  if (path.startsWith(ROUTES.img)) return "img";
  if (path.startsWith(ROUTES.deploy)) return "deploy";
  if (path.startsWith(ROUTES.analytics)) return "analytics";
  if (path === ROUTES.gallery) return "gallery";
  return "create";
}

function chromeForPath(path) {
  if (path.startsWith(`${ROUTES.bat}/batches/`)) return { product: "ForgeBAT", breadcrumbs: [{ label: "Batches", path: ROUTES.bat }, { label: `Batch ${path.split("/").at(-1)}` }] };
  if (path === `${ROUTES.bat}/new`) return { product: "ForgeBAT", breadcrumbs: [{ label: "Batches", path: ROUTES.bat }, { label: "New batch" }] };
  if (path.startsWith(ROUTES.bat)) return { product: "ForgeBAT", breadcrumbs: [{ label: "Batches" }] };
  if (path.startsWith(ROUTES.vid)) return { product: "ForgeVID", breadcrumbs: [{ label: "Motion" }] };
  if (path.startsWith(ROUTES.img)) return { product: "ForgeIMG", breadcrumbs: [{ label: "Transform" }] };
  if (path.startsWith(ROUTES.deploy)) {
    const parts = path.split("/").filter(Boolean);
    const breadcrumbs = [{ label: "Pipelines", path: parts.length > 1 ? ROUTES.deploy : null }];
    if (parts[1] === "pipelines" && parts[2]) breadcrumbs.push({ label: DEPLOY_ROUTE_LABELS[parts[2]] ?? parts[2].replaceAll("_", " ") });
    if (parts[1] === "jobs" && parts[2]) breadcrumbs.push({ label: `#${parts[2]}` });
    return { product: "ForgeDeploy", breadcrumbs };
  }
  if (path.startsWith(ROUTES.analytics)) return { product: "ForgeAnalytics", breadcrumbs: [{ label: "Analysis" }] };
  return { product: "ForgeAI", breadcrumbs: [{ label: path === ROUTES.gallery ? "Gallery" : "Create" }] };
}

function AutoTextarea({ value, onChange, ...props }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    ref.current.style.height = "140px";
    ref.current.style.height = `${Math.min(Math.max(ref.current.scrollHeight, 140), 240)}px`;
  }, [value]);
  return (
    <textarea
      ref={ref}
      value={value}
      onChange={onChange}
      className="forge-input min-h-[140px] resize-y leading-[1.55] scrollbar-subtle"
      {...props}
    />
  );
}

function AccountDialog({ open, initialMode, registrationOpen, returningUser, authenticatorAvailable, authenticatorName, connected, required, onClose, onAuthenticate, onSelectWorkspace, onContinueAsGuest }) {
  const wasOpen = useRef(false);
  const submitRef = useRef(null);
  const submittingRef = useRef(false);
  const otpRefs = useRef([]);
  const [mode, setMode] = useState(initialMode);
  const [step, setStep] = useState("credentials");
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [recoveryMode, setRecoveryMode] = useState(false);
  const [visible, setVisible] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [authenticatedUser, setAuthenticatedUser] = useState(null);
  const [openingWorkspace, setOpeningWorkspace] = useState(null);
  const [activeTheme, setActiveTheme] = useState(() => readNyxTheme());
  const [totpSeconds, setTotpSeconds] = useState(() => 30 - (Math.floor(Date.now() / 1000) % 30));

  useEffect(() => {
    if (open) {
      if (!wasOpen.current) {
        setStep("credentials");
        setMode(authenticatorAvailable ? "authenticator" : returningUser ? "login" : initialMode === "register" && !registrationOpen ? "login" : initialMode);
        setEmail(returningUser?.email ?? "");
        setAuthenticatedUser(null);
        setError("");
      }
      wasOpen.current = true;
      return;
    }
    wasOpen.current = false;
    setDisplayName("");
    setEmail("");
    setPassword("");
    setCode("");
    setRecoveryMode(false);
    setVisible(false);
    setAuthenticatedUser(null);
    setOpeningWorkspace(null);
  }, [open, initialMode, registrationOpen, returningUser, authenticatorAvailable]);

  useEffect(() => {
    const updateTheme = (event) => setActiveTheme(event.detail.theme);
    window.addEventListener("nyxforge:theme", updateTheme);
    return () => window.removeEventListener("nyxforge:theme", updateTheme);
  }, []);

  useEffect(() => {
    if (mode !== "authenticator" || recoveryMode || submitting || code.length !== 6) return;
    submitRef.current();
  }, [code, mode, recoveryMode, submitting]);

  useEffect(() => {
    if (!open || mode !== "authenticator" || recoveryMode) return undefined;
    const updateCountdown = () => setTotpSeconds(30 - (Math.floor(Date.now() / 1000) % 30));
    updateCountdown();
    const timer = window.setInterval(updateCountdown, 1000);
    return () => window.clearInterval(timer);
  }, [open, mode, recoveryMode]);

  const submit = async (event) => {
    event?.preventDefault();
    if (submittingRef.current) return;
    if (mode === "authenticator" && !recoveryMode && code.length !== 6) {
      setError("Enter all six digits from your authenticator app.");
      otpRefs.current[Math.min(code.length, 5)]?.focus();
      return;
    }
    submittingRef.current = true;
    setSubmitting(true);
    setError("");
    try {
      const session = await onAuthenticate(mode, { displayName, email, password, code });
      setAuthenticatedUser(session.user);
      setPassword("");
      setCode("");
      setStep("workspace");
    } catch (requestError) {
      setError(requestError.message);
      if (mode === "authenticator") {
        setCode("");
        if (!recoveryMode) window.requestAnimationFrame(() => otpRefs.current[0]?.focus());
      }
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  };
  submitRef.current = submit;

  const setAuthenticatorMode = () => {
    setMode("authenticator");
    setRecoveryMode(false);
    setPassword("");
    setCode("");
    setError("");
    window.requestAnimationFrame(() => otpRefs.current[0]?.focus());
  };

  const setRecoveryCodeMode = () => {
    setMode("authenticator");
    setRecoveryMode(true);
    setCode("");
    setError("");
  };

  const setPasswordMode = () => {
    setMode("login");
    setRecoveryMode(false);
    setCode("");
    setError("");
  };

  const enterWorkspace = async (workspace) => {
    if (openingWorkspace) return;
    setOpeningWorkspace(workspace);
    setSubmitting(true);
    setError("");
    try {
      await onSelectWorkspace(workspace);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setOpeningWorkspace(null);
      setSubmitting(false);
    }
  };

  const updateOtpDigit = (index, value) => {
    const digits = value.replace(/\D/g, "");
    if (digits.length > 1) {
      const pastedCode = digits.slice(0, 6);
      setCode(pastedCode);
      otpRefs.current[Math.min(pastedCode.length, 6) - 1]?.focus();
      return;
    }
    if (!digits) {
      setCode((current) => current.slice(0, index));
      return;
    }
    setCode((current) => `${current.slice(0, index)}${digits}${current.slice(index + 1)}`.slice(0, 6));
    otpRefs.current[index + 1]?.focus();
  };

  const handleOtpKeyDown = (index, event) => {
    if (event.key === "Backspace" && !code[index] && index > 0) {
      event.preventDefault();
      setCode((current) => current.slice(0, index - 1));
      otpRefs.current[index - 1]?.focus();
    } else if (event.key === "ArrowLeft" && index > 0) {
      event.preventDefault();
      otpRefs.current[index - 1]?.focus();
    } else if (event.key === "ArrowRight" && index < 5) {
      event.preventDefault();
      otpRefs.current[index + 1]?.focus();
    }
  };

  const welcomeName = authenticatorName ?? returningUser?.display_name;
  const dialogTitle = step === "workspace"
    ? `Where are we creating today${authenticatedUser?.display_name ? `, ${authenticatedUser.display_name}` : ""}?`
    : mode === "authenticator"
      ? `Welcome back${welcomeName ? `, ${welcomeName}` : ""}`
      : returningUser ? `Welcome back, ${returningUser.display_name}` : mode === "register" ? "Create admin account" : "Admin sign in";
  const dialogDescription = step === "workspace"
    ? "Choose one creator workspace for this signed-in session. Delivery and analytics remain shared."
    : mode === "authenticator"
      ? recoveryMode ? "Enter one of your saved recovery codes." : "Enter the six-digit code from your authenticator app."
      : returningUser ? "Enter your password to unlock the Forge suite." : "Sign in to access the Forge suite.";
  return (
    <Dialog open={open} onClose={submitting || required || step === "workspace" ? () => {} : onClose} className="relative z-[110]">
      <DialogBackdrop className="nyx-auth-backdrop fixed inset-0" />
      <div className="nyx-auth-screen fixed inset-0 overflow-y-auto">
        <header className="nyx-auth-header"><div className="nyx-brand"><span className="nyx-brand-mark">N</span><strong>NYXFORGE</strong><span className="nyx-product-badge">v1.0—FORGE</span></div><div className={`nyx-auth-core ${connected ? "is-online" : "is-offline"}`}><i />{step === "workspace" ? <>{connected ? "ALL SYSTEMS CONNECTED" : "SYSTEMS DEGRADED"} <span>/</span> <span className="nyx-auth-core-meta"><b>SESSION:</b> ADMIN AUTHENTICATED</span> <span>/</span> <span className="nyx-auth-core-meta"><b>NODE:</b> LOCAL</span></> : mode === "authenticator" ? <>FORGE CORE {connected ? "99.99%" : "OFFLINE"} <span>/</span> SESSION: FACTOR 1 OF 2 <span>/</span> NODE: LOCAL</> : <>REFORGE {connected ? "ONLINE" : "UNAVAILABLE"} <span>/</span> SESSION: UNAUTHENTICATED</>}</div><div className="nyx-auth-theme"><span>THEME</span><NyxThemePicker showLabel={step === "workspace"} /></div></header>
        <div className={`nyx-auth-layout ${step === "workspace" ? "is-workspace" : mode === "authenticator" ? "is-two-factor" : ""}`}>
          {step !== "workspace" && <section className="nyx-auth-intro"><span className="nyx-auth-signal"><i />{mode === "authenticator" ? "Second factor required" : "Forge suite access"}</span><h1>{mode === "authenticator" ? <>Welcome back,<br />{welcomeName || "Admin"}</> : mode === "register" ? <>Create admin<br />account</> : <>Admin<br />sign in</>}</h1><p>{dialogDescription}</p><div className="nyx-auth-stats">{mode === "authenticator" ? <><span className="is-online"><small>Factor 1</small><b>Verified</b></span><span><small>Device</small><b>{recoveryMode ? "Recovery" : "Authenticator"}</b></span><span><small>Code expires</small><b>00:{String(totpSeconds).padStart(2, "0")}</b></span></> : <><span className={connected ? "is-online" : "is-offline"}><small>reForge</small><b>{connected ? "Online" : "Unavailable"}</b></span><span><small>Channel</small><b>{window.location.protocol === "https:" ? "HTTPS" : "Local HTTP"}</b></span><span><small>Access</small><b>Admin</b></span></>}</div></section>}
          <DialogPanel className={`w-full ${step === "workspace" ? "nyx-workspace-panel" : `nyx-auth-panel ${mode === "authenticator" ? "is-two-factor" : ""}`}`}>
            {step !== "workspace" && <div className="nyx-dialog-bar"><span><b>&gt;_</b> AUTH TERMINAL</span><i>SECURE</i></div>}
            {step === "credentials" && mode === "authenticator" && <nav className="nyx-auth-modes" aria-label="Sign-in method">
              <button type="button" aria-selected={!recoveryMode} onClick={setAuthenticatorMode}>Authenticator</button>
              <button type="button" aria-selected={recoveryMode} onClick={setRecoveryCodeMode}>Recovery</button>
              <button type="button" aria-selected="false" onClick={setPasswordMode}>Password</button>
            </nav>}
            {!(step === "credentials" && mode === "authenticator") && <div className={step === "workspace" ? "nyx-workspace-intro" : "flex items-start justify-between gap-4"}>
              <div>
                {step === "workspace" && <p className="workspace-eyebrow"><i />Select workspace</p>}
                <DialogTitle className="text-[18px] font-extrabold tracking-[-0.01em] text-[var(--text)]">{dialogTitle}</DialogTitle>
                {step !== "workspace" && <p className="mt-[6px] text-[13.5px] font-medium leading-[1.5] text-[var(--text-muted)]">{dialogDescription}</p>}
              </div>
              {step === "workspace" && <p>{dialogDescription}</p>}
              {!required && step === "credentials" && <button type="button" onClick={onClose} disabled={submitting} className="nyx-close" aria-label="Close account dialog"><X size={16} /></button>}
            </div>}

            {step === "credentials" ? <>{mode !== "authenticator" && !returningUser && <div className={`forge-seg mt-[18px] ${registrationOpen ? "grid grid-cols-2" : "grid grid-cols-1"}`}>
              {[["login", "Admin sign in"], ...(registrationOpen ? [["register", "Create account"]] : [])].map(([value, label]) => (
                <button key={value} type="button" onClick={() => { setMode(value); setError(""); }} aria-selected={mode === value}>{label}</button>
              ))}
            </div>}

            <form onSubmit={submit} className={mode === "authenticator" ? "nyx-two-factor-form" : "mt-5 space-y-4"}>
              {mode === "authenticator" ? (
                recoveryMode ? <label className="nyx-two-factor-field">Recovery code
                  <input autoFocus type="text" autoComplete="one-time-code" value={code} onChange={(event) => setCode(event.target.value.toUpperCase())} minLength={16} maxLength={19} required className="forge-input" placeholder="XXXX-XXXX-XXXX-XXXX" />
                </label> : <div className="nyx-two-factor-field"><span>Authenticator code</span>
                  <div className="nyx-otp-grid" role="group" aria-label="Six-digit authenticator code">
                    {Array.from({ length: 6 }, (_, index) => <input key={index} ref={(element) => { otpRefs.current[index] = element; }} autoFocus={index === 0} type="text" inputMode="numeric" autoComplete={index === 0 ? "one-time-code" : "off"} value={code[index] ?? ""} onChange={(event) => updateOtpDigit(index, event.target.value)} onKeyDown={(event) => handleOtpKeyDown(index, event)} onPaste={(event) => { const pastedCode = event.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6); if (!pastedCode) return; event.preventDefault(); setCode(pastedCode); otpRefs.current[Math.min(pastedCode.length, 6) - 1]?.focus(); }} maxLength={6} aria-label={`Digit ${index + 1}`} disabled={submitting} />)}
                  </div>
                  <small>{submitting ? "Verifying encrypted code…" : `${code.length} / 6 digits entered`}</small>
                </div>
              ) : mode === "register" && (
                <label className="block text-xs font-medium text-zinc-300">Display name
                  <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} minLength={2} maxLength={50} required autoComplete="name" className="forge-input mt-2 bg-[var(--well)]" placeholder="How should we address you?" />
                </label>
              )}
              {mode !== "authenticator" && !returningUser && <label className="block text-xs font-medium text-zinc-300">Email
                <input autoFocus={mode === "login"} type="email" value={email} onChange={(event) => setEmail(event.target.value)} required autoComplete="email" className="forge-input mt-2 bg-[var(--well)]" placeholder="you@example.com" />
              </label>}
              {mode !== "authenticator" && <label className="block text-xs font-medium text-zinc-300">Password
                <div className="relative mt-2">
                  <input autoFocus={Boolean(returningUser)} type={visible ? "text" : "password"} value={password} onChange={(event) => setPassword(event.target.value)} minLength={10} maxLength={256} required autoComplete={mode === "register" ? "new-password" : "current-password"} className="forge-input bg-[var(--well)] pr-11" placeholder={mode === "register" ? "At least 10 characters" : "Your password"} />
                  <button type="button" onClick={() => setVisible((current) => !current)} className="absolute inset-y-0 right-0 grid w-11 place-items-center text-zinc-500 transition hover:text-white" aria-label={visible ? "Hide password" : "Show password"}>{visible ? <EyeOff size={16} /> : <Eye size={16} />}</button>
                </div>
              </label>}
              {mode === "login" && authenticatorAvailable && <button type="button" onClick={setAuthenticatorMode} className="w-full text-center text-[11.5px] font-semibold text-[var(--accent-hi)] hover:text-[var(--accent-text)]">Use authenticator instead</button>}
              {error && <p className="rounded-[9px] bg-[var(--danger-wash)] px-[13px] py-[10px] text-[12px] leading-[1.45] text-[#ff9d9d]">{error}</p>}
              {(mode !== "authenticator" || recoveryMode) && <button type="submit" disabled={submitting} className="forge-btn-primary w-full">
                {!submitting && (mode === "register" ? <UserPlus size={16} /> : mode === "authenticator" ? <ShieldCheck size={16} /> : <LogIn size={16} />)}
                {submitting ? "Validating..." : mode === "register" ? "Create account and continue" : "Validate and continue"}
              </button>}
              {mode === "authenticator" && !recoveryMode && <button type="submit" disabled={submitting} className="nyx-two-factor-submit">{submitting ? "Verifying…" : "Verify and continue"}</button>}
              {mode === "authenticator" && <div className="nyx-two-factor-alternatives">
                <button type="button" onClick={recoveryMode ? setAuthenticatorMode : setRecoveryCodeMode}>{recoveryMode ? "Use authenticator code" : "Use a recovery code"}</button>
                <button type="button" onClick={setPasswordMode}>Use password instead</button>
              </div>}
              {onContinueAsGuest && mode === "authenticator" && <div className="nyx-two-factor-divider"><span>Or</span></div>}
              {onContinueAsGuest && <button type="button" disabled={submitting} onClick={onContinueAsGuest} className={mode === "authenticator" ? "nyx-two-factor-guest" : "w-full text-center text-[11.5px] font-semibold text-[var(--text-muted)] transition hover:text-[var(--text)] disabled:opacity-60"}>Continue as guest</button>}
            </form></> : <div className="workspace-grid">
              {WORKSPACES.map(({ id, label, code: workspaceCode, category, description }) => <article key={id} className={`workspace-card ${id === "forgeai" ? "is-primary" : ""}`}>
                <header><span>{workspaceCode}</span><strong>{label}</strong><em>{category}</em></header>
                <div><p>{description}</p><button type="button" disabled={submitting} onClick={() => enterWorkspace(id)}>{openingWorkspace === id ? <><LoaderCircle size={14} className="animate-spin" /> Opening {label}</> : <>Enter {label}</>}</button></div>
              </article>)}
              {error && <p className="workspace-message is-error">{error}</p>}
            </div>}
            {step === "credentials" && mode === "authenticator" && <div className="nyx-two-factor-channel"><span>TLS 1.3 encrypted channel</span><span>{submitting ? "Verifying" : "Awaiting code"}<i /></span></div>}
          </DialogPanel>
        </div>
        <footer className="nyx-auth-footer"><span className="nyx-status-mode">&gt;_ {step === "workspace" ? "WORKSPACE" : mode === "authenticator" ? "2FA" : "AUTH"}</span><span><b>SUITE:</b> FORGE</span>{step === "workspace" && <span><b>THEME:</b> <em>{NYX_THEMES.find((theme) => theme.id === activeTheme)?.label ?? "Solar"}</em></span>}{mode === "authenticator" && step === "credentials" && <button type="button" onClick={setPasswordMode}>&lt; BACK TO SIGN IN</button>}<span className="ml-auto">{step === "workspace" ? <><b>DELIVERY + ANALYTICS:</b> SHARED</> : "GUEST ACCESS: AVAILABLE"}</span></footer>
      </div>
    </Dialog>
  );
}

function WorkspaceSwitcherDialog({ open, activeWorkspace, operationsSummary, onClose, onSelect }) {
  const [switchingTo, setSwitchingTo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) {
      setSwitchingTo(null);
      setError("");
    }
  }, [open]);

  const choose = async (workspace) => {
    if (workspace === activeWorkspace) {
      onClose();
      return;
    }
    setSwitchingTo(workspace);
    setError("");
    try {
      await onSelect(workspace);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSwitchingTo(null);
    }
  };

  return (
    <Dialog open={open} onClose={switchingTo ? () => {} : onClose} className="relative z-[125]">
      <DialogBackdrop className="nyx-switch-backdrop fixed inset-0" />
      <div className="nyx-switch-screen fixed inset-0 overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4 sm:p-8">
          <DialogPanel className="nyx-switch-panel w-full max-w-[760px]">
            <div className="nyx-switch-titlebar">
              <span><b>&gt;_</b> Switch workspace</span>
              <button type="button" onClick={onClose} disabled={Boolean(switchingTo)} className="nyx-close" aria-label="Close workspace switcher"><X size={16} /></button>
            </div>
            <div className="nyx-switch-content">
              <div className="nyx-switch-intro">
                <DialogTitle>Where would you like to work?</DialogTitle>
                <p><i />You stay signed in. Running and queued work continues in the background while you switch.</p>
              </div>

              <div className="nyx-switch-grid">
              {WORKSPACES.map(({ id, label, description }) => {
                const current = id === activeWorkspace;
                const busy = switchingTo === id;
                const content = <><span className="nyx-switch-copy"><span><strong>{label}</strong>{current && <em>Current workspace</em>}</span><small>{description}</small></span>{!current && <span className="nyx-switch-action">{busy ? <LoaderCircle size={15} className="animate-spin" /> : `Open ${label}`}</span>}</>;
                return current
                  ? <div key={id} className="nyx-switch-card is-current">{content}</div>
                  : <button key={id} type="button" disabled={Boolean(switchingTo)} onClick={() => choose(id)} className="nyx-switch-card">{content}</button>;
              })}
              </div>

              {error && <p className="nyx-switch-error">{error}</p>}
              <div className="nyx-switch-footer">
                <span><i /><b>Operations</b>{operationsSummary?.active ? `${operationsSummary.running} running · ${operationsSummary.queued} queued` : "None running or queued"}</span>
                <button type="button" onClick={onClose} disabled={Boolean(switchingTo)}>Stay here</button>
              </div>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

function ControlDialogShell({
  open,
  busy = false,
  onClose,
  title,
  eyebrow,
  description,
  status = "READY",
  maxWidth = "max-w-[760px]",
  children,
  footer,
}) {
  return (
    <Dialog open={open} onClose={busy ? () => {} : onClose} className="relative z-[120]">
      <DialogBackdrop className="nyx-control-backdrop fixed inset-0" />
      <div className="nyx-control-screen fixed inset-0 overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4 sm:p-8">
          <DialogPanel className={`nyx-control-panel w-full ${maxWidth}`}>
            <div className="nyx-control-titlebar">
              <span><b>&gt;_</b> {title}</span>
              <div>
                <i className={busy ? "is-busy" : ""}>{busy ? "WORKING" : status}</i>
                <button type="button" onClick={onClose} disabled={busy} className="nyx-close" aria-label={`Close ${title}`}><X size={16} /></button>
              </div>
            </div>
            <div className="nyx-control-content">
              <header className="nyx-control-heading">
                <span><i />{eyebrow}</span>
                <DialogTitle>{title}</DialogTitle>
                <p>{description}</p>
              </header>
              {children}
            </div>
            {footer && <footer className="nyx-control-footer">{footer}</footer>}
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

function OperationToast({ toast, onDismiss, onNavigate }) {
  useEffect(() => {
    const timer = window.setTimeout(() => onDismiss(toast.id), 7000);
    return () => window.clearTimeout(timer);
  }, [onDismiss, toast.id]);
  const toneClass = toast.tone === "danger" ? "text-[var(--danger)]" : toast.tone === "success" ? "text-[var(--ok)]" : "text-[var(--text-ui)]";
  return <div role={toast.tone === "danger" ? "alert" : "status"} className="pointer-events-auto w-full rounded-[12px] bg-[var(--field)] p-4 shadow-[var(--shadow-menu)] ring-1 ring-inset ring-[var(--divider)]">
    <div className="flex items-start gap-3"><span className={`mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-[8px] bg-[var(--raised)] ${toneClass}`}>{toast.tone === "success" ? <Check size={14} /> : toast.tone === "danger" ? <X size={14} /> : <Layers size={14} />}</span><div className="min-w-0 flex-1"><p className={`text-[12.5px] font-extrabold ${toneClass}`}>{toast.title}</p><p className="mt-1 line-clamp-2 text-[11.5px] font-medium leading-[1.45] text-[var(--text-muted)]">{toast.detail}</p><button type="button" onClick={() => { onNavigate(ROUTES.deploy); onDismiss(toast.id); }} className="mt-2 text-[11px] font-bold text-[var(--accent-hi)] hover:text-[var(--accent-text)]">View operations →</button></div><button type="button" onClick={() => onDismiss(toast.id)} className="rounded-md p-1 text-[var(--text-subtle)] hover:bg-[var(--raised)] hover:text-[var(--text)]" aria-label="Dismiss notification"><X size={13} /></button></div>
  </div>;
}

function OperationToastViewport({ toasts, onDismiss, onNavigate }) {
  if (!toasts.length) return null;
  return <aside aria-label="Operation notifications" className="pointer-events-none fixed right-4 top-[72px] z-[140] flex w-[min(360px,calc(100vw-32px))] flex-col gap-2">{toasts.map((toast) => <OperationToast key={toast.id} toast={toast} onDismiss={onDismiss} onNavigate={onNavigate} />)}</aside>;
}

function AuthenticatorSetupDialog({ open, enabled, onClose, onComplete }) {
  const [step, setStep] = useState("password");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [enrollment, setEnrollment] = useState(null);
  const [recoveryCodes, setRecoveryCodes] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (open) return;
    setStep("password");
    setPassword("");
    setCode("");
    setEnrollment(null);
    setRecoveryCodes([]);
    setBusy(false);
    setError("");
  }, [open]);

  const start = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      setEnrollment(await beginAuthenticatorSetup(password));
      setPassword("");
      setStep("scan");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };

  const confirm = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await confirmAuthenticatorSetup(code);
      setRecoveryCodes(result.recovery_codes);
      setStep("recovery");
      onComplete(result.user);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };

  const description = step === "password"
    ? `${enabled ? "Replace" : "Set up"} Google Authenticator or another TOTP app.`
    : step === "scan"
      ? "Scan the QR code, then enter the current six-digit code."
      : "Save these one-time recovery codes somewhere private.";
  return (
    <ControlDialogShell open={open} busy={busy} onClose={onClose} title="Authenticator sign-in" eyebrow="Account security" description={description} status={step === "recovery" ? "ENABLED" : "SECURE"} maxWidth="max-w-[620px]">
      <div className="nyx-control-stepbar">
        {[["password", "01 Verify"], ["scan", "02 Pair"], ["recovery", "03 Recover"]].map(([id, label]) => <span key={id} className={step === id ? "is-active" : ""}>{label}</span>)}
      </div>
      {step === "password" && <form onSubmit={start} className="nyx-control-section space-y-4">
        <div className="nyx-control-section-title"><Smartphone size={16} /><div><strong>Confirm identity</strong><small>Your password authorizes changes to two-factor security.</small></div></div>
        <label className="nyx-control-label">Current password<input autoFocus type="password" value={password} onChange={(event) => setPassword(event.target.value)} minLength={10} maxLength={256} required autoComplete="current-password" className="forge-input" /></label>
        {error && <p className="nyx-control-error">{error}</p>}
        <button type="submit" disabled={busy} className="forge-btn-primary w-full">{busy ? <><LoaderCircle size={15} className="animate-spin" /> Preparing</> : <><ShieldCheck size={15} /> Show QR code</>}</button>
      </form>}
      {step === "scan" && enrollment && <form onSubmit={confirm} className="nyx-control-section space-y-4">
        <div className="nyx-authenticator-pairing"><div className="nyx-authenticator-qr"><img src={enrollment.qr_data_url} alt="Authenticator enrollment QR code" /></div><div className="min-w-0"><p className="forge-eyebrow">Manual setup key</p><code>{enrollment.manual_secret}</code><p>Use this key only when your authenticator cannot scan the QR code.</p></div></div>
        <label className="nyx-control-label">Six-digit code<input autoFocus type="text" inputMode="numeric" autoComplete="one-time-code" value={code} onChange={(event) => setCode(event.target.value.replace(/\D/g, "").slice(0, 6))} minLength={6} maxLength={6} required className="forge-input text-center font-mono text-[22px] font-bold tracking-[0.28em]" placeholder="000000" /></label>
        {error && <p className="nyx-control-error">{error}</p>}
        <button type="submit" disabled={busy || code.length !== 6} className="forge-btn-primary w-full">{busy ? <><LoaderCircle size={15} className="animate-spin" /> Verifying</> : <><Check size={15} /> Verify and enable</>}</button>
      </form>}
      {step === "recovery" && <section className="nyx-control-section space-y-4">
        <div className="nyx-control-section-title"><ShieldCheck size={16} /><div><strong>Authenticator enabled</strong><small>Store these one-time codes outside this computer.</small></div></div>
        <div className="nyx-recovery-grid">{recoveryCodes.map((recoveryCode) => <code key={recoveryCode}>{recoveryCode}</code>)}</div>
        <p className="nyx-control-note">Each code works once if your phone is unavailable. They will not be shown again.</p>
        <div className="nyx-control-actions"><button type="button" onClick={() => navigator.clipboard.writeText(recoveryCodes.join("\n"))} className="forge-btn flex-1"><Copy size={15} /> Copy codes</button><button type="button" onClick={onClose} className="forge-btn-primary flex-1"><Check size={16} /> Codes saved</button></div>
      </section>}
    </ControlDialogShell>
  );
}

function CloudSettingsDialog({ open, onClose, credentials, onSave }) {
  const [draftKeys, setDraftKeys] = useState(() => cloudProviderState(""));
  const [visible, setVisible] = useState(() => cloudProviderState(false));
  const [editing, setEditing] = useState(() => cloudProviderState(false));
  const [testing, setTesting] = useState("");
  const [errors, setErrors] = useState(() => cloudProviderState(""));

  useEffect(() => {
    if (open) return;
    setDraftKeys(cloudProviderState(""));
    setVisible(cloudProviderState(false));
    setEditing(cloudProviderState(false));
    setErrors(cloudProviderState(""));
  }, [open]);

  const saveProvider = async (provider) => {
    const apiKey = draftKeys[provider].trim();
    if (!apiKey) {
      setErrors((current) => ({ ...current, [provider]: "Enter an API key first." }));
      return;
    }
    setTesting(provider);
    setErrors((current) => ({ ...current, [provider]: "" }));
    try {
      await onSave(provider, apiKey);
      setDraftKeys((current) => ({ ...current, [provider]: "" }));
      setVisible((current) => ({ ...current, [provider]: false }));
      setEditing((current) => ({ ...current, [provider]: false }));
    } catch (error) {
      setErrors((current) => ({ ...current, [provider]: error.message }));
    } finally {
      setTesting("");
    }
  };

  return (
    <ControlDialogShell open={open} busy={Boolean(testing)} onClose={onClose} title="Cloud API settings" eyebrow="Prompt intelligence" description="Connect optional cloud prompt providers. Keys remain in this browser session and are never stored in the ForgeAI database." status="SESSION ONLY" maxWidth="max-w-[800px]">
            <div className="nyx-control-grid sm:grid-cols-2">
              {CLOUD_PROVIDERS.map((provider) => {
                const configured = Boolean(credentials[provider.value]?.apiKey);
                const showInput = !configured || editing[provider.value];
                return (
                  <section key={provider.value} className="nyx-control-section">
                    <div className="nyx-control-section-title">
                      <KeyRound size={16} />
                      <div><strong>{provider.label}</strong><small>Prompt generation provider</small></div>
                      {configured && !showInput && <span className="nyx-control-status is-online"><i />Configured</span>}
                    </div>

                    {showInput ? (
                      <>
                        <label className="nyx-control-label mt-4">API key</label>
                        <div className="relative">
                          <input
                            type={visible[provider.value] ? "text" : "password"}
                            value={draftKeys[provider.value]}
                            onChange={(event) => setDraftKeys((current) => ({ ...current, [provider.value]: event.target.value }))}
                            placeholder={provider.keyHint}
                            autoComplete="new-password"
                            spellCheck="false"
                            className="forge-input pr-10 font-mono text-[11.5px]"
                          />
                          <button type="button" onClick={() => setVisible((current) => ({ ...current, [provider.value]: !current[provider.value] }))} className="absolute inset-y-0 right-0 grid w-10 place-items-center text-[var(--nyx-label)] transition hover:text-[var(--nyx-signal)]" aria-label={visible[provider.value] ? "Hide API key" : "Show API key"}>
                            {visible[provider.value] ? <EyeOff size={15} /> : <Eye size={15} />}
                          </button>
                        </div>
                        {errors[provider.value] && <p className="nyx-control-error mt-3">{errors[provider.value]}</p>}
                        <div className="nyx-control-actions mt-3">
                          {configured && <button type="button" onClick={() => setEditing((current) => ({ ...current, [provider.value]: false }))} className="forge-btn flex-1">Cancel</button>}
                          <button type="button" onClick={() => saveProvider(provider.value)} disabled={testing === provider.value} className="forge-btn-primary flex-1 px-3 py-[10px] text-[12px]">
                            {testing === provider.value ? <><LoaderCircle size={14} className="animate-spin" /> Testing</> : <><ShieldCheck size={14} /> Test & save</>}
                          </button>
                        </div>
                      </>
                    ) : (
                      <div className="mt-4">
                        <p className="nyx-control-note">Validated for {credentials[provider.value].models.length} available model{credentials[provider.value].models.length === 1 ? "" : "s"}. The key is hidden and cannot be recovered.</p>
                        <button type="button" onClick={() => setEditing((current) => ({ ...current, [provider.value]: true }))} className="forge-btn mt-3 w-full">Replace key</button>
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
            <p className="nyx-control-note is-warning">Cloud models may incur provider charges. ForgeAI sends only prompt instructions and active generation settings.</p>
    </ControlDialogShell>
  );
}

function Sidebar({ form, setForm, onOpenPromptPicker, models, modelProfiles, promptModels, ollamaConnected, cloudCredentials, surpriseNotice, connected, loading, surpriseLoading, onSurprise, onGenerate, onSimpleGenerate, account }) {
  const update = (key) => (event) => setForm((current) => ({
    ...current,
    [key]: event.target.value,
    ...(["prompt", "negative_prompt"].includes(key) ? { prompt_run_id: null } : {}),
  }));
  const activeProfile = modelProfiles[form.model];
  // A chosen reuse prompt skips generation, so the controls that only steer
  // generation have no effect on the render.
  const promptIsReused = form.prompt_engine === "reuse" && Boolean(form.prompt.trim());
  const qualitySummary = qualityOutputSummary(form.resolution_mode, form.aspect_ratio);
  const configuredCloudProviders = account?.is_admin
    ? CLOUD_PROVIDERS.filter(({ value }) => cloudCredentials[value]?.apiKey)
    : [];
  const cloudProviderOptions = configuredCloudProviders.map(({ value, label }) => ({ value, label }));
  const activeCloud = cloudCredentials[form.cloud_provider];
  const modelOptions = models.map((model) => ({
    value: model,
    label: modelProfiles[model]?.display_name ?? model,
    description: modelProfiles[model]?.category_label ?? "Forge checkpoint",
  }));
  const styles = activeProfile?.styles ?? ["Photoreal"];
  const minCfg = activeProfile?.min_cfg_scale ?? 3;
  const ratioOptions = activeProfile?.aspect_ratios?.length ? activeProfile.aspect_ratios : RATIOS;
  const generationRatings = RATINGS;
  const chooseModel = (model) => {
    const profile = modelProfiles[model];
    const modelRatios = profile?.aspect_ratios?.length ? profile.aspect_ratios : RATIOS;
    setForm((current) => ({
      ...current,
      model,
      prompt_run_id: null,
      style: profile?.styles.includes(current.style) ? current.style : (profile?.default_style ?? "Photoreal"),
      // A CFG carried over from a low-floor model (Lightning/Hyper) would
      // otherwise starve a full-step checkpoint.
      cfg_scale: current.cfg_scale === null ? null : Math.max(current.cfg_scale, profile?.min_cfg_scale ?? 3),
      aspect_ratio: modelRatios.includes(current.aspect_ratio) ? current.aspect_ratio : modelRatios[0],
    }));
  };
  return (
    <section className="nyx-create-panel">
      <div className="nyx-panel-titlebar">
        <span><b>&gt;_</b> Creation panel</span>
        <div className="forge-seg nyx-panel-mode" role="tablist" aria-label="Interface mode">
          {[['simple', 'Simple'], ['advanced', 'Advanced']].map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={form.ui_mode === value}
              onClick={() => setForm((current) => ({ ...current, ui_mode: value }))}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="nyx-create-scroll scrollbar-subtle">
        <section className="nyx-model-card">
        <div className="mb-4 flex items-center justify-between">
          <label className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Model</label>
          <span
            className={`inline-flex items-center gap-1.5 rounded-[20px] px-[9px] py-[3px] text-[10.5px] font-bold ${connected ? "bg-[var(--ok-wash)] text-[var(--ok)]" : "bg-[var(--idle-wash)] text-[var(--idle)]"}`}
            title={connected ? "Forge connected" : "Forge unavailable — launch with --api"}
          >
            <span className={`h-[7px] w-[7px] rounded-full ${connected ? "dot-live bg-[var(--ok)]" : "bg-[var(--idle)]"}`} />
            {connected ? "Connected" : "Offline"}
          </span>
        </div>
        <div className="mb-4">
          <CustomSelect value={form.model} options={modelOptions} onChange={chooseModel} ariaLabel="Model" />
          {activeProfile?.description && <p className="mt-2 text-[10px] leading-4 text-zinc-500">{activeProfile.description}</p>}
          {activeProfile?.settings_summary && <p className="mt-1 text-[10px] leading-4 text-zinc-600">{form.cfg_scale === null ? `Auto preset: ${activeProfile.settings_summary}` : `Preset: ${activeProfile.settings_summary.replace(/CFG [0-9.]+/, `CFG ${form.cfg_scale}`)} (CFG overridden)`}</p>}
        </div>
        </section>

        <div className="nyx-style-control">
          <div className="mb-2 flex items-center justify-between"><span className="text-[11px] font-medium text-zinc-400">Style</span><span className="text-[10px] text-zinc-600">{form.style}</span></div>
        <div className="mb-6 flex flex-wrap gap-2">
          {styles.map((style) => (
            <button
              key={style}
              type="button"
              onClick={() => setForm((current) => ({ ...current, style, prompt_run_id: null }))}
              className={`rounded-[20px] px-[13px] py-[7px] text-[12px] font-semibold transition-colors ${form.style === style ? "bg-[var(--accent-wash)] text-[var(--accent-text)]" : "bg-[var(--field)] text-[var(--text-2)] hover:bg-[var(--field-hover)]"}`}
            >
              {style}
            </button>
          ))}
        </div>
        </div>

        <section className="nyx-creation-controls space-y-4">
          {form.ui_mode === "advanced" && (
            <>
              <label className="block text-xs font-medium text-zinc-300">
                <span className="mb-2 block">Positive prompt</span>
                <AutoTextarea value={form.prompt} onChange={update("prompt")} placeholder="Describe the subject, setting, lighting, composition, and mood…" />
              </label>
              <label className="block text-xs font-medium text-zinc-300">
                <span className="mb-2 block">Negative prompt</span>
                <AutoTextarea value={form.negative_prompt} onChange={update("negative_prompt")} placeholder="Optional: unwanted objects, styles, colors, or details…" />
              </label>
            </>
          )}
          <div>
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-300">CFG scale — prompt adherence</span>
              <span className="text-[10px] text-zinc-600">
                {form.cfg_scale === null ? `Auto (${activeProfile?.cfg_scale ?? "model preset"})` : `${form.cfg_scale} · ${form.cfg_scale <= minCfg ? "weakest prompt hold" : form.cfg_scale >= 8 ? "rigid, over-baked" : form.cfg_scale <= minCfg + 1.5 ? "softer, most natural skin" : "balanced"}`}
              </span>
            </div>
            <input
              type="range"
              min={minCfg}
              max="12"
              step="0.5"
              value={form.cfg_scale ?? activeProfile?.cfg_scale ?? 6}
              onChange={(event) => setForm((current) => ({ ...current, cfg_scale: Number(event.target.value) }))}
              className="h-1 w-full accent-[var(--accent)]"
              aria-label="CFG scale"
            />
            <div className="nyx-range-labels"><span>Loose {minCfg}</span><span>Strict 12</span></div>
            {form.cfg_scale !== null && (
              <button
                type="button"
                onClick={() => setForm((current) => ({ ...current, cfg_scale: null }))}
                className="mt-1 text-[10px] text-zinc-500 transition hover:text-zinc-300"
              >
                Reset to model preset
              </button>
            )}
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-300">Prompt intelligence</span>
              <span className={`text-[10px] ${(form.prompt_engine === "cloud" ? configuredCloudProviders.length > 0 : form.prompt_engine === "ollama" ? ollamaConnected : true) ? "text-[var(--accent-hi)]" : "text-[var(--text-faint)]"}`}>
                {form.prompt_engine === "reuse"
                  ? form.prompt.trim() ? `From #${form.prompt_source_id ?? "?"}` : "None chosen"
                  : form.prompt_engine === "cloud"
                    ? configuredCloudProviders.length > 0 ? `${configuredCloudProviders.length} cloud provider${configuredCloudProviders.length === 1 ? "" : "s"}` : "Cloud not configured"
                    : form.prompt_engine === "ollama"
                      ? ollamaConnected ? "Ollama ready" : "Ollama offline"
                      : "Local rules"}
              </span>
            </div>
            <div className="forge-seg grid grid-cols-4">
              {[["curated", "Curated"], ["ollama", "Ollama"], ["cloud", "Cloud API"], ["reuse", "Reuse"]].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  disabled={(value === "ollama" && !promptModels.length) || (value === "cloud" && !account?.is_admin)}
                  onClick={() => setForm((current) => ({ ...current, prompt_engine: value, prompt_run_id: null, ...(value === "reuse" ? {} : { prompt: "", negative_prompt: "", prompt_source_id: null }) }))}
                  aria-selected={form.prompt_engine === value}
                  className="disabled:cursor-not-allowed disabled:text-[var(--text-disabled)]"
                >
                  {label}
                </button>
              ))}
            </div>
            {form.prompt_engine === "reuse" && (
              <div className="mt-2">
                {form.prompt.trim() ? (
                  <>
                    <div title={form.prompt} className="flex items-center gap-2 rounded-[9px] bg-[var(--well)] px-2.5 py-2">
                      <span className="text-[11px] font-bold tabular-nums text-[var(--accent-text)]">#{form.prompt_source_id ?? "—"}</span>
                      <span className="truncate text-[10.5px] text-[var(--text-muted)]">{form.prompt}</span>
                    </div>
                    <div className="mt-2 flex gap-2">
                      <button type="button" onClick={() => onOpenPromptPicker()} className="flex-1 rounded-[8px] bg-[var(--field)] px-2 py-2 text-[10.5px] font-bold text-[var(--text-ui)] transition hover:bg-[var(--field-hover)]">Change</button>
                      <button type="button" onClick={() => setForm((current) => ({ ...current, prompt: "", negative_prompt: "", prompt_run_id: null, prompt_source_id: null }))} className="flex-1 rounded-[8px] bg-[var(--field)] px-2 py-2 text-[10.5px] font-bold text-[var(--text-muted)] transition hover:text-[var(--text-ui)]">Clear</button>
                    </div>
                  </>
                ) : (
                  <button type="button" onClick={() => onOpenPromptPicker()} className="w-full rounded-[9px] bg-[var(--accent-wash)] px-2 py-2.5 text-[10.5px] font-bold text-[var(--accent-text)] ring-1 ring-[var(--accent)] transition hover:bg-[var(--field-hover)]">Choose a past prompt…</button>
                )}
              </div>
            )}
            {form.prompt_engine === "ollama" && promptModels.length > 0 && (
              <div className="mt-2">
                <CustomSelect value={form.ollama_model} options={promptModels} onChange={(ollama_model) => setForm((current) => ({ ...current, ollama_model, prompt_run_id: null }))} ariaLabel="Ollama prompt model" />
              </div>
            )}
            {form.prompt_engine === "cloud" && (
              configuredCloudProviders.length ? (
                <div className="mt-2 space-y-2">
                  <CustomSelect
                    value={form.cloud_provider}
                    options={cloudProviderOptions}
                    onChange={(cloud_provider) => {
                      const credential = cloudCredentials[cloud_provider];
                      setForm((current) => ({
                        ...current,
                        cloud_provider,
                        cloud_model: credential?.selectedModel ?? credential?.recommended ?? credential?.models?.[0] ?? "",
                        prompt_run_id: null,
                      }));
                    }}
                    ariaLabel="Cloud provider"
                  />
                  {activeCloud?.models?.length > 0 && (
                    <CustomSelect value={form.cloud_model} options={activeCloud.models} onChange={(cloud_model) => setForm((current) => ({ ...current, cloud_model, prompt_run_id: null }))} ariaLabel="Cloud prompt model" />
                  )}
                </div>
              ) : (
                <div className="mt-2 rounded-xl bg-[#17181f] px-3 py-2.5 text-[11px] leading-5 text-zinc-500">Add a provider from the avatar menu under <span className="font-medium text-zinc-300">API configuration</span>.</div>
              )
            )}
            <div className="nyx-direction-grid">
            <div className={`mt-3${promptIsReused ? " opacity-50" : ""}`}>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-medium text-zinc-400">Orientation</span>
                <span className="text-[10px] text-zinc-600">
                  {promptIsReused ? `Set by prompt #${form.prompt_source_id ?? "?"}` : form.orientation === "front" ? "Facing the camera" : form.orientation === "side" ? "Side and profile" : form.orientation === "back" ? "Rear and over-shoulder" : "All angles"}
                </span>
              </div>
              <div className="forge-seg grid grid-cols-4" role="radiogroup" aria-label="Body orientation">
                {[["mixed", "Mixed"], ["front", "Front"], ["side", "Side"], ["back", "Back"]].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    role="radio"
                    aria-checked={form.orientation === value}
                    disabled={promptIsReused}
                    onClick={() => setForm((current) => ({ ...current, orientation: value, prompt_run_id: null }))}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className={`mt-3${promptIsReused ? " opacity-50" : ""}`}>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-medium text-zinc-400">Creativity</span>
                <span className="text-[10px] text-zinc-600">
                  {promptIsReused ? `Set by prompt #${form.prompt_source_id ?? "?"}` : form.creativity_level === "consistent" ? "Favors familiar winners" : form.creativity_level === "experimental" ? "Maximum scene variety" : "Fresh but dependable"}
                </span>
              </div>
              <div className="forge-seg grid grid-cols-3" role="radiogroup" aria-label="Prompt creativity">
                {[["consistent", "Consistent"], ["balanced", "Balanced"], ["experimental", "Explore"]].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    role="radio"
                    aria-checked={form.creativity_level === value}
                    disabled={promptIsReused}
                    onClick={() => setForm((current) => ({ ...current, creativity_level: value, prompt_run_id: null }))}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            </div>
            {surpriseNotice && <p className="mt-2 text-[10px] leading-4 text-zinc-500">{surpriseNotice}</p>}
          </div>
          {form.ui_mode === "advanced" && (
            <button
              type="button"
              onClick={onSurprise}
              disabled={surpriseLoading}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#1f2028] px-3 py-2.5 text-xs font-semibold text-zinc-200 transition hover:bg-zinc-700 hover:text-white disabled:cursor-wait disabled:text-zinc-500"
            >
              {surpriseLoading ? <LoaderCircle size={15} className="animate-spin" /> : <WandSparkles size={15} />}
              {surpriseLoading ? "Creating prompt…" : "Auto-generate prompt"}
            </button>
          )}
        </section>

        <div className="my-6 h-px bg-[var(--divider)]" />
        <div className="nyx-output-controls">
          <label className="block text-xs font-medium text-zinc-300">Aspect ratio
            <div className="mt-2"><CustomSelect value={form.aspect_ratio} options={ratioOptions} onChange={(aspect_ratio) => setForm((current) => ({ ...current, aspect_ratio, prompt_run_id: null }))} ariaLabel="Aspect ratio" /></div>
          </label>
          <label className="block text-xs font-medium text-zinc-300">Content rating
            <div className="mt-2"><CustomSelect value={form.content_rating} options={generationRatings} onChange={(content_rating) => setForm((current) => ({ ...current, content_rating, prompt_run_id: null }))} ariaLabel="Content rating" /></div>
          </label>
          <div>
            <span className="mb-2 block text-xs font-medium text-zinc-300">Output quality</span>
            <div className="forge-seg grid grid-cols-3" role="radiogroup" aria-label="Output quality">
              {QUALITY_MODES.map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={form.resolution_mode === value}
                  onClick={() => setForm((current) => ({ ...current, resolution_mode: value, prompt_run_id: null }))}
                >
                  {label}
                </button>
              ))}
            </div>
            <p className="mt-2 text-[11px] leading-4 text-zinc-500">
              {qualitySummary}
            </p>
          </div>
        </div>
      <div className="nyx-create-submit">
        <button
          onClick={form.ui_mode === "simple" ? onSimpleGenerate : onGenerate}
          disabled={loading}
          className="forge-btn-primary w-full"
        >
          {!loading && <Sparkles size={17} />}
          {loading ? "Adding to queue…" : form.ui_mode === "simple" ? "Create surprise" : "Generate image"}
        </button>
        <p>{loading ? "Your request is being added to the shared queue." : form.ui_mode === "simple" ? "ForgeAI builds the prompt from these controls and submits the job." : "Your prompt and settings will be submitted to the execution queue."}</p>
      </div>
      </div>
    </section>
  );
}

function RatingStatus({ image }) {
  const rated = image.user_rating != null;
  return (
    <span
      className={`grid h-7 min-w-7 place-items-center rounded-[6px] bg-[rgba(12,10,17,0.72)] px-1.5 ${rated ? "text-[var(--accent-hi)]" : "text-[var(--text-2)]"}`}
      title={rated ? `Rated ${image.user_rating} out of 5` : "Not rated"}
      aria-label={rated ? `Rated ${image.user_rating} out of 5` : "Not rated"}
    >
      <Star size={14} fill={rated ? "currentColor" : "none"} />
    </span>
  );
}

function ThumbnailDownload({ image, className = "" }) {
  return (
    <a
      href={image.full_url}
      download={`nyx-forge-${image.id}.png`}
      onClick={(event) => event.stopPropagation()}
      className={`grid h-7 w-7 place-items-center rounded-[6px] bg-[rgba(12,10,17,0.72)] text-[var(--text-2)] transition-colors hover:bg-[var(--accent-wash)] hover:text-[var(--accent-text)] ${className}`}
      title="Download image"
      aria-label="Download image"
    >
      <Download size={13} />
    </a>
  );
}

function QueueSettingsDialog({ open, account, onClose, onSave }) {
  const [value, setValue] = useState(account?.max_active_jobs ?? 3);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [storage, setStorage] = useState(null);
  const [storagePath, setStoragePath] = useState("");
  const [storageSaving, setStorageSaving] = useState(false);
  const [storageError, setStorageError] = useState("");
  const [storageSaved, setStorageSaved] = useState(false);

  useEffect(() => {
    if (open) {
      setValue(account?.max_active_jobs ?? 3);
      setError("");
      setStorageError("");
      setStorageSaved(false);
      getStorageSettings()
        .then((state) => {
          setStorage(state);
          setStoragePath(state.configured_dir ?? state.active_dir ?? "");
        })
        .catch((requestError) => setStorageError(requestError.message));
    }
  }, [account?.max_active_jobs, open]);

  const saveStorage = async () => {
    setStorageSaving(true);
    setStorageError("");
    try {
      const state = await setStorageSettings(storagePath.trim());
      setStorage(state);
      setStorageSaved(true);
    } catch (requestError) {
      setStorageError(requestError.message);
    } finally {
      setStorageSaving(false);
    }
  };

  const submit = async () => {
    setSaving(true);
    setError("");
    try {
      await onSave(value);
      onClose();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSaving(false);
    }
  };

  return <ControlDialogShell
    open={open}
    busy={saving || storageSaving}
    onClose={onClose}
    title="Settings"
    eyebrow="Local configuration"
    description="Control queue capacity and the local storage used by every Forge workspace."
    status="LOCAL"
    maxWidth="max-w-[700px]"
    footer={<><span><b>Scope</b> Account + local host</span><div><button type="button" onClick={onClose} disabled={saving || storageSaving} className="forge-btn">Close</button><button type="button" onClick={submit} disabled={saving || storageSaving} className="forge-btn-primary">{saving ? <><LoaderCircle size={14} className="animate-spin" /> Saving</> : <><Save size={14} /> Save queue limit</>}</button></div></>}
  >
        <section className="nyx-control-section">
          <div className="nyx-control-section-title"><Layers size={16} /><div><strong>Execution queue</strong><small>Account preference</small></div></div>
          <label className="nyx-control-label mt-4">Maximum outstanding jobs<div><CustomSelect value={value} options={[1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50].map((item) => ({ value: item, label: `${item} job${item === 1 ? "" : "s"}` }))} onChange={(next) => setValue(Number(next))} ariaLabel="Maximum outstanding jobs" /></div></label>
          <p className="nyx-control-note mt-3">Forge renders one image at a time on the GPU. Prompt preparation can run concurrently; rendering jobs wait in FIFO order.</p>
          {error && <p className="nyx-control-error mt-3">{error}</p>}
        </section>

        <section className="nyx-control-section">
          <div className="nyx-control-section-title"><Save size={16} /><div><strong>Storage folder</strong><small>Local host path</small></div></div>
          <p className="nyx-control-note mt-4">Generated images, uploads, and the database are written here.</p>
          <div className="mt-3 flex gap-2">
            <input
              value={storagePath}
              onChange={(event) => { setStoragePath(event.target.value); setStorageSaved(false); }}
              disabled={storageSaving || storage?.env_override}
              spellCheck={false}
              placeholder="D:\nyx-forge-data"
              aria-label="Storage folder path"
              className="forge-input min-w-0 flex-1 font-mono text-[11.5px]"
            />
            <button type="button" onClick={saveStorage} disabled={storageSaving || storage?.env_override || !storagePath.trim()} className="forge-btn shrink-0">{storageSaving ? <LoaderCircle size={14} className="animate-spin" /> : <Save size={14} />}{storageSaving ? "Saving" : "Apply path"}</button>
          </div>
          {storage && (
            <p className="nyx-control-path mt-3">
              <span>Active path</span><code>{storage.active_dir}</code>
              {storage.free_gb != null && ` · ${storage.free_gb} GB free`}
            </p>
          )}
          {storage?.env_override && <p className="nyx-control-note is-warning mt-3">NYX_FORGE_DATA_DIR overrides this setting. Clear the environment variable before choosing a folder here.</p>}
          {(storageSaved || storage?.pending_restart) && !storage?.env_override && <p className="nyx-control-note is-success mt-3">Saved. Restart the app to activate this folder. Existing images are not moved automatically.</p>}
          {storageError && <p className="nyx-control-error mt-3">{storageError}</p>}
        </section>
  </ControlDialogShell>;
}

function BackendSettingsDialog({ open, backends, onClose, onRefresh, onStatus }) {
  const [drafts, setDrafts] = useState({});
  const [busy, setBusy] = useState({});
  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (!open) return;
    setErrors({});
    onRefresh();
    getBackendSettings()
      .then((settings) => setDrafts(Object.fromEntries(settings.map((backend) => [backend.id, { port: backend.port, package_dir: backend.package_dir }]))))
      .catch((requestError) => setErrors({ _load: requestError.message }));
  }, [open]); // Refresh once when the dialog opens; the global poll keeps it current.

  useEffect(() => {
    if (!open) return;
    setDrafts((current) => {
      const updated = { ...current };
      backends.forEach((backend) => {
        if (!updated[backend.id]) updated[backend.id] = { port: backend.port, package_dir: "" };
      });
      return updated;
    });
  }, [backends, open]);

  const run = async (backend, operation) => {
    setBusy((current) => ({ ...current, [backend.id]: operation }));
    setErrors((current) => ({ ...current, [backend.id]: "" }));
    try {
      const next = operation === "save"
        ? await updateBackend(backend.id, Number(drafts[backend.id]?.port), drafts[backend.id]?.package_dir.trim())
        : await controlBackend(backend.id, operation);
      onStatus(next);
    } catch (requestError) {
      setErrors((current) => ({ ...current, [backend.id]: requestError.message }));
    } finally {
      setBusy((current) => ({ ...current, [backend.id]: "" }));
      onRefresh();
    }
  };

  const isBusy = Object.values(busy).some(Boolean);
  const onlineCount = backends.filter((backend) => backend.reachable).length;
  return <ControlDialogShell open={open} busy={isBusy} onClose={onClose} title="Forge backends" eyebrow="Local model runners" description="Configure and control the Stability Matrix Forge installations used across every workspace." status={`${onlineCount}/${backends.length} ONLINE`} maxWidth="max-w-[900px]">
        <div className="nyx-control-grid md:grid-cols-2">
          {errors._load && <p className="nyx-control-error md:col-span-2">{errors._load}</p>}
          {backends.map((backend) => {
            const draft = drafts[backend.id] ?? { port: backend.port, package_dir: "" };
            const operation = busy[backend.id];
            const stateLabel = backend.reachable ? "Reachable" : backend.running ? "Starting" : "Stopped";
            const stateColor = backend.reachable ? "var(--ok)" : backend.running ? "var(--warn)" : "var(--danger)";
            return <section key={backend.id} className="nyx-control-section">
              <div className="nyx-control-section-title"><RotateCw size={16} /><div><strong>{backend.label}</strong><small>{backend.id}</small></div><span className={`nyx-control-status ${backend.reachable ? "is-online" : backend.running ? "is-busy" : "is-offline"}`}><i style={{ backgroundColor: stateColor }} />{stateLabel}</span></div>
              <div className="mt-4 grid gap-3 sm:grid-cols-[120px_1fr]">
                <label className="nyx-control-label">Port<input type="number" min="1" max="65535" value={draft.port} onChange={(event) => setDrafts((current) => ({ ...current, [backend.id]: { ...draft, port: event.target.value } }))} className="forge-input font-mono text-[11.5px]" /></label>
                <label className="nyx-control-label">Package folder<input value={draft.package_dir} onChange={(event) => setDrafts((current) => ({ ...current, [backend.id]: { ...draft, package_dir: event.target.value } }))} spellCheck={false} className="forge-input font-mono text-[11.5px]" /></label>
              </div>
              {backend.loaded_checkpoint && <p className="nyx-control-path mt-3" title={backend.loaded_checkpoint}><span>Loaded checkpoint</span><code>{backend.loaded_checkpoint}</code></p>}
              {(backend.error || errors[backend.id]) && <p className="nyx-control-error mt-3">{errors[backend.id] || backend.error}</p>}
              <div className="nyx-control-actions mt-4">
                <button type="button" onClick={() => run(backend, "save")} disabled={Boolean(operation) || !draft.package_dir.trim() || !Number(draft.port)} className="forge-btn"><Save size={13} />{operation === "save" ? "Saving" : "Save"}</button>
                <button type="button" onClick={() => run(backend, "start")} disabled={Boolean(operation) || backend.running} className="nyx-control-action is-start"><Play size={13} />{operation === "start" ? "Starting" : "Start"}</button>
                <button type="button" onClick={() => run(backend, "stop")} disabled={Boolean(operation) || !backend.running} className="nyx-control-action is-stop"><Square size={12} />{operation === "stop" ? "Stopping" : "Stop"}</button>
                <button type="button" onClick={() => run(backend, "restart")} disabled={Boolean(operation) || !backend.running} className="forge-btn"><RotateCw size={13} />{operation === "restart" ? "Restarting" : "Restart"}</button>
              </div>
            </section>;
          })}
        </div>
  </ControlDialogShell>;
}

function GalleryActionsDialog({ image, saving, upscalingId, onClose, onRate, onUpscale, onPixelUpscale }) {
  const [draftScore, setDraftScore] = useState(0);
  const [draftReasons, setDraftReasons] = useState(() => new Set());
  const lowScoreNeedsReason = draftScore > 0 && draftScore <= 2 && draftReasons.size === 0;
  const busy = upscalingId != null;
  const alreadyDetailRendered = Boolean(image?.high_res || image?.super_res);

  useEffect(() => {
    setDraftScore(image?.user_rating ?? 0);
    setDraftReasons(new Set(image?.feedback_reasons ?? []));
  }, [image?.id, image?.user_rating, image?.feedback_reasons]);

  const toggleReason = (reason) => {
    setDraftReasons((current) => {
      const updated = new Set(current);
      if (updated.has(reason)) updated.delete(reason);
      else updated.add(reason);
      return updated;
    });
  };

  const launchUpscale = async (kind, value) => {
    if (!image || busy) return;
    if (kind === "pixel") await onPixelUpscale(image.id, value);
    else await onUpscale(image.id, value);
  };

  return (
    <Dialog open={Boolean(image)} onClose={saving ? () => {} : onClose} className="relative z-[125]">
      <DialogBackdrop className="nyx-gallery-actions-backdrop fixed inset-0" />
      <div className="nyx-gallery-actions-screen fixed inset-0 overflow-y-auto p-4 sm:p-8">
        <div className="flex min-h-full items-center justify-center">
          <DialogPanel className="nyx-gallery-actions-panel w-full max-w-[680px] overflow-hidden">
            <header className="nyx-gallery-actions-titlebar">
              <DialogTitle><b>&gt;_</b> Image actions</DialogTitle>
              <span>{image ? `Generation #${image.id}` : "No selection"}</span>
              <button type="button" onClick={onClose} disabled={saving} aria-label="Close image actions"><X size={15} /></button>
            </header>

            {image && <>
              <div className="nyx-gallery-actions-meta">
                <span><small>Style</small><strong>{displayStyle(image)}</strong></span>
                <span><small>Quality</small><strong>{displayQuality(image)}</strong></span>
                <span><small>Output</small><strong>{image.width} × {image.height}</strong></span>
                <span><small>Status</small><strong>Stored locally</strong></span>
              </div>
              <div className="nyx-gallery-actions-body scrollbar-subtle">
                <section className="nyx-gallery-action-section">
                  <div className="nyx-gallery-action-heading"><span><b>01</b> / Feedback</span>{image.user_rating && <i>Current {image.user_rating}/5</i>}</div>
                  <h3>Rate this result</h3>
                  <p>Feedback improves future scene and prompt choices.</p>
                  <div className="nyx-gallery-stars" aria-label="Image quality rating">
                    {[1, 2, 3, 4, 5].map((score) => <button key={score} type="button" disabled={saving} onClick={() => setDraftScore(score)} aria-label={`${score} out of 5`} aria-pressed={score <= draftScore}><Star size={21} fill={score <= draftScore ? "currentColor" : "none"} /></button>)}
                  </div>
                  {draftScore > 0 && <div className="nyx-gallery-feedback">
                    <p>What influenced it? <span className={draftScore <= 2 ? "is-required" : ""}>{draftScore <= 2 ? "Choose at least one" : "Optional"}</span></p>
                    <div className="nyx-gallery-reasons">{FEEDBACK_REASONS.map((reason) => {
                      const selected = draftReasons.has(reason.code);
                      return <button key={reason.code} type="button" aria-pressed={selected} onClick={() => toggleReason(reason.code)}>{draftScore >= 4 ? reason.positive : reason.negative}</button>;
                    })}</div>
                    <button type="button" disabled={saving || lowScoreNeedsReason} onClick={() => onRate(image.id, draftScore, [...draftReasons])} className="nyx-gallery-save-rating">{saving ? "Saving…" : image.user_rating ? "Update rating" : "Save rating"}</button>
                  </div>}
                </section>

                <section className="nyx-gallery-action-section">
                  <div className="nyx-gallery-action-heading"><span><b>02</b> / Resolution</span>{upscalingId === image.id && <i className="is-busy"><LoaderCircle size={11} className="animate-spin" /> Running</i>}</div>
                  <h3>Upscale this image</h3>
                  <p>Choose a pixel resize or spend GPU time on a new detail pass.</p>
                  <label>Pixel resize <span>/ no re-render</span></label>
                  <div className="nyx-gallery-upscale-grid is-pixel">{[1.5, 2, 3, 4].map((multiplier) => <button key={multiplier} type="button" disabled={busy} onClick={() => launchUpscale("pixel", multiplier)}>{multiplier}×</button>)}</div>
                  <label>Detail pass <span>/ GPU re-render</span></label>
                  {alreadyDetailRendered && <p className="nyx-gallery-action-warning">This image is already detail-rendered. Another latent pass can redraw composition or anatomy; Pixel Resize is safer.</p>}
                  <div className="nyx-gallery-upscale-grid is-detail">
                    <button type="button" disabled={busy} onClick={() => launchUpscale("detail", false)}><Sparkles size={13} /> High detail</button>
                    <button type="button" disabled={busy} onClick={() => launchUpscale("detail", true)}><WandSparkles size={13} /> Super detail</button>
                  </div>
                </section>
              </div>
              <footer className="nyx-gallery-actions-footer">
                <span><i /> Local artifact secured</span>
                {image.job_id != null && <a href={`/forge-deploy/jobs/${image.job_id}`} target="_blank" rel="noopener noreferrer">Open deployment <ExternalLink size={12} /></a>}
              </footer>
            </>}
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

function VideoGallery({ onShowImages }) {
  const [videos, setVideos] = useState([]); const [cursor, setCursor] = useState(null); const [loading, setLoading] = useState(true); const [active, setActive] = useState(null); const [error, setError] = useState("");
  const load = useCallback(async (before = null) => { setLoading(true); setError(""); try { const page = await getVideoHistory(before); setVideos((current) => before ? [...current, ...page.items] : page.items); setCursor(page.next_cursor); } catch (requestError) { setError(requestError.message); } finally { setLoading(false); } }, []);
  useEffect(() => { load(); }, [load]);
  return <main className="forge-page px-8 py-7"><section className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between"><div><div className="flex items-center gap-3"><h1 className="forge-display">Gallery</h1><div className="forge-seg grid grid-cols-2"><button onClick={onShowImages}>Images</button><button aria-selected="true">Videos</button></div></div><p className="forge-body mt-2">Play and revisit every ForgeVID creation.</p></div><span className="w-fit rounded-full bg-[var(--accent-wash)] px-3 py-1 text-[11px] font-bold text-[var(--accent-text)]">{videos.length} videos</span></section><ErrorBanner error={error} onDismiss={()=>setError("")} className="mb-4"/>
    {videos.length ? <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">{videos.map((video)=><article key={video.id} className="overflow-hidden rounded-[14px] bg-[var(--panel)] ring-1 ring-inset ring-[var(--divider)]"><button onClick={()=>setActive(video)} className="relative grid aspect-video w-full place-items-center overflow-hidden bg-[var(--well)]"><video src={video.file_url} preload="metadata" muted className="h-full w-full object-cover"/><span className="absolute grid h-11 w-11 place-items-center rounded-full bg-[var(--scrim)] text-white"><Play size={18} fill="currentColor"/></span><span className="absolute bottom-2 right-2 rounded-md bg-[var(--scrim)] px-2 py-1 text-[10px] font-bold text-white">{video.duration_seconds.toFixed(1)}s</span></button><div className="flex items-start justify-between gap-3 p-4"><div className="min-w-0"><p className="line-clamp-2 text-[12.5px] font-semibold text-[var(--text)]">{video.prompt}</p><p className="mt-1 text-[10.5px] text-[var(--text-muted)]">{video.mode.toUpperCase()} · {video.width}×{video.height}</p></div><button aria-label="Toggle favourite" onClick={async()=>{try{const updated=await setVideoFavorite(video.id,!video.favorite);setVideos((items)=>items.map((item)=>item.id===video.id?updated:item));}catch(e){setError(e.message);}}} className={video.favorite?"text-[var(--danger)]":"text-[var(--text-muted)]"}><Heart size={16} fill={video.favorite?"currentColor":"none"}/></button></div></article>)}</div> : <div className="grid min-h-[50vh] place-items-center rounded-[14px] bg-[var(--well)] text-[13px] font-semibold text-[var(--text-muted)]">{loading?"Loading videos…":"No videos yet. Create one in ForgeVID."}</div>}{cursor&&<div className="mt-7 text-center"><button className="forge-btn" disabled={loading} onClick={()=>load(cursor)}>{loading?"Loading…":"Load more"}</button></div>}
    <Dialog open={Boolean(active)} onClose={()=>setActive(null)} className="relative z-[110]"><DialogBackdrop className="fixed inset-0 bg-[var(--scrim)]"/><div className="fixed inset-0 grid place-items-center p-5"><DialogPanel className="w-full max-w-[1100px] rounded-[16px] bg-[var(--field)] p-4 shadow-[var(--shadow-modal)]"><div className="mb-3 flex items-center justify-between"><DialogTitle className="text-[14px] font-bold text-[var(--text)]">ForgeVID preview</DialogTitle><button onClick={()=>setActive(null)} className="rounded-lg p-2 text-[var(--text-muted)] hover:bg-[var(--raised)]"><X size={18}/></button></div>{active&&<video src={active.file_url} controls preload="metadata" className="max-h-[78vh] w-full rounded-[11px] bg-black"/>}</DialogPanel></div></Dialog></main>;
}

function GalleryPage({ images, selectedIndex, onChoose, onToggleFavorite, hasMore, loadingMore, onLoadMore, error, ratingImageId, ratingSaving, onOpenRating, onDismissRating, onRate, upscalingId, onUpscale, onPixelUpscale }) {
  const [mediaMode, setMediaMode] = useState("images");
  const [styleFilter, setStyleFilter] = useState("All styles");
  const [ratingFilter, setRatingFilter] = useState("All ratings");
  const [orientationFilter, setOrientationFilter] = useState("All orientations");
  const [favoritesOnly, setFavoritesOnly] = useState(false);

  const orientation = (image) => {
    if (image.width === image.height) return "Square";
    return image.width > image.height ? "Landscape" : "Portrait";
  };
  const filtered = images
    .map((image, index) => ({ image, index }))
    .filter(({ image }) => styleFilter === "All styles" || image.style === styleFilter)
    .filter(({ image }) => ratingFilter === "All ratings" || image.rating === ratingFilter)
    .filter(({ image }) => orientationFilter === "All orientations" || orientation(image) === orientationFilter)
    .filter(({ image }) => !favoritesOnly || image.favorite);
  const galleryStyles = [...new Set(images.map((image) => image.style))].sort();
  const actionImage = images.find((image) => image.id === ratingImageId) ?? null;
  if (mediaMode === "videos") return <VideoGallery onShowImages={()=>setMediaMode("images")}/>;

  return (
    <main className="nyx-gallery forge-page px-8 py-7">
      <section className="nyx-gallery-head mb-[26px] flex flex-col gap-[18px] lg:flex-row lg:items-end lg:justify-between">
        <div className="flex flex-col gap-[6px]">
          <div className="flex items-center gap-[10px]"><h1 className="forge-display">Gallery</h1><div className="forge-seg grid grid-cols-2"><button aria-selected="true">Images</button><button onClick={()=>setMediaMode("videos")}>Videos</button></div><span className="nyx-gallery-count rounded-[20px] bg-[var(--accent-wash)] px-[9px] py-[3px] text-[10.5px] font-bold text-[var(--accent-text)] tabular-nums">{loadingMore && <LoaderCircle size={11} className="animate-spin" />}{loadingMore ? "Syncing latest" : `${images.length} generations`}</span></div>
          <p className="forge-body">Browse, filter, and open any ForgeAI creation.</p>
        </div>
        <div className="flex w-full items-center gap-3 lg:w-[780px]">
          <button type="button" aria-pressed={favoritesOnly} onClick={() => setFavoritesOnly((value) => !value)} title="Show favourites only" className={`flex h-[42px] shrink-0 items-center gap-2 rounded-[10px] px-3.5 text-[12px] font-bold transition ${favoritesOnly ? "bg-[var(--accent-wash)] text-[var(--accent-text)] ring-1 ring-[var(--accent)]" : "bg-[var(--field)] text-[var(--text-muted)] hover:text-[var(--text-ui)]"}`}>
            <Heart size={14} fill={favoritesOnly ? "currentColor" : "none"} />
            Favourites
          </button>
          <div className="grid min-w-0 flex-1 gap-3 sm:grid-cols-3">
          <CustomSelect value={styleFilter} options={["All styles", ...galleryStyles]} onChange={setStyleFilter} ariaLabel="Filter by style" />
          <CustomSelect value={ratingFilter} options={["All ratings", ...RATINGS]} onChange={setRatingFilter} ariaLabel="Filter by rating" />
          <CustomSelect value={orientationFilter} options={["All orientations", "Portrait", "Landscape", "Square"]} onChange={setOrientationFilter} ariaLabel="Filter by orientation" />
          </div>
        </div>
      </section>
      <ErrorBanner error={error} onDismiss={() => setError("")} className="mb-4" />

      {filtered.length ? (
        <PhotoSwipeGallery entries={filtered} selectedIndex={selectedIndex} onSelect={onChoose} className="nyx-gallery-grid grid">
          {({ image, index, ref, open }) => (
            <div
              key={image.id}
              className={`nyx-gallery-card group relative overflow-hidden bg-[var(--panel)] text-left transition duration-[120ms] ${selectedIndex === index ? "is-selected" : ""}`}
            >
              <button
                ref={ref}
                type="button"
                onClick={open}
                className="absolute inset-0 h-full w-full overflow-hidden rounded-[13px] text-left"
              >
              <img
                src={image.thumbnail_url}
                loading="lazy"
                decoding="async"
                alt={image.positive_prompt}
                className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
              />
              <div className="absolute inset-x-0 bottom-0 translate-y-2 bg-gradient-to-t from-black/95 via-black/70 to-transparent px-4 pb-3 pt-12 opacity-0 transition duration-200 group-hover:translate-y-0 group-hover:opacity-100">
                <p className="truncate text-sm font-medium text-white">{displayStyle(image)} · {displayQuality(image)}</p>
                <p className="mt-0.5 text-xs text-zinc-400">{orientation(image)}</p>
              </div>
              </button>
              <span className="absolute left-2 top-2 z-20 rounded-[7px] bg-[rgba(12,10,17,0.82)] px-2 py-1 text-[10.5px] font-bold tabular-nums text-[var(--text-ui)] backdrop-blur">#{image.id}</span>
              <button type="button" aria-label={image.favorite ? "Remove from favourites" : "Add to favourites"} aria-pressed={Boolean(image.favorite)} onClick={(event) => { event.stopPropagation(); onToggleFavorite(image); }} className={`absolute bottom-2 right-11 z-20 grid h-8 w-8 place-items-center rounded-[8px] bg-[rgba(12,10,17,0.82)] backdrop-blur transition hover:bg-[var(--accent-wash)] ${image.favorite ? "text-[#ff6b8a]" : "text-[var(--text-muted)] hover:text-[var(--accent-text)]"}`}><Heart size={14} fill={image.favorite ? "currentColor" : "none"} /></button>
              <button type="button" onClick={() => onOpenRating(image.id)} className="absolute right-2 top-2 z-20 flex h-8 items-center gap-1.5 rounded-[8px] bg-[rgba(12,10,17,0.82)] px-2.5 text-[11.5px] font-bold text-[var(--text-ui)] backdrop-blur transition hover:bg-[var(--accent-wash)] hover:text-[var(--accent-text)]"><WandSparkles size={13} /> {image.user_rating ? `${image.user_rating}/5 · Actions` : "Actions"}</button>
              <ThumbnailDownload image={image} className="absolute bottom-2 right-2 z-20 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100" />
            </div>
          )}
        </PhotoSwipeGallery>
      ) : (
        <div className="grid min-h-[55vh] place-items-center rounded-[14px] bg-[var(--well)]">
          <div className="text-center"><h3 className="text-[14.5px] font-bold text-[var(--text-muted)]">No matching images</h3><p className="mt-2 text-[12.5px] font-semibold text-[var(--accent-hi)]">Change one of the gallery filters.</p></div>
        </div>
      )}
      {hasMore && (
        <div className="mt-8 flex justify-center">
          <button
            type="button"
            onClick={onLoadMore}
            disabled={loadingMore}
            className="forge-btn min-w-36"
          >
            {loadingMore && <LoaderCircle size={16} className="animate-spin" />}
            {loadingMore ? "Loading..." : "Load more"}
          </button>
        </div>
      )}
      <GalleryActionsDialog image={actionImage} saving={ratingSaving} upscalingId={upscalingId} onClose={onDismissRating} onRate={onRate} onUpscale={onUpscale} onPixelUpscale={onPixelUpscale} />
    </main>
  );
}

function RatingControl({ image, expanded, saving, disabled, onExpand, onRate, onDismiss }) {
  const [draftScore, setDraftScore] = useState(image.user_rating ?? 0);
  const [draftReasons, setDraftReasons] = useState(() => new Set(image.feedback_reasons ?? []));
  const controlRef = useRef(null);
  const lowScoreNeedsReason = draftScore > 0 && draftScore <= 2 && draftReasons.size === 0;

  useEffect(() => {
    setDraftScore(image.user_rating ?? 0);
    setDraftReasons(new Set(image.feedback_reasons ?? []));
  }, [image.id, image.user_rating, image.feedback_reasons]);

  useEffect(() => {
    if (!expanded) return undefined;
    const closeOnOutsideClick = (event) => {
      if (!controlRef.current?.contains(event.target)) onDismiss();
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") onDismiss();
    };
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [expanded, onDismiss]);

  const toggleReason = (reason) => {
    setDraftReasons((current) => {
      const updated = new Set(current);
      if (updated.has(reason)) updated.delete(reason);
      else updated.add(reason);
      return updated;
    });
  };

  return (
    <div ref={controlRef} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={expanded ? onDismiss : onExpand}
        aria-expanded={expanded}
        title={disabled ? "Reveal the image before rating it" : image.user_rating ? "Edit image rating" : "Rate this image"}
        className="group flex h-9 items-center gap-2 rounded-[9px] bg-[rgba(18,16,26,0.92)] px-3 text-xs font-semibold text-[var(--text-ui)] backdrop-blur transition-colors hover:bg-[var(--field-hover)] disabled:cursor-not-allowed disabled:opacity-45"
      >
        <Star size={16} className={image.user_rating ? "fill-current" : "transition group-hover:fill-current"} />
        {image.user_rating ? `${image.user_rating}/5` : "Rate"}
      </button>
      {expanded && !disabled && (
        <div className="absolute right-0 top-11 z-30 max-h-[calc(100vh-12rem)] w-[min(26rem,calc(100vw-3rem))] overflow-y-auto rounded-[12px] bg-[var(--field)] p-4 text-left shadow-[var(--shadow-menu)]" role="dialog" aria-label="Rate this generated image">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-semibold text-white">How did this turn out?</p>
              <p className="mt-0.5 text-[11px] text-zinc-500">Your feedback improves future scene and prompt choices.</p>
            </div>
            <button type="button" onClick={onDismiss} className="grid h-7 w-7 shrink-0 place-items-center rounded-lg text-zinc-500 transition hover:bg-white/[0.07] hover:text-white" aria-label="Close rating panel">
              <X size={15} />
            </button>
          </div>
          <div className="mt-3 flex items-center gap-1" aria-label="Image quality rating">
            {[1, 2, 3, 4, 5].map((score) => (
              <button key={score} type="button" disabled={saving} onClick={() => setDraftScore(score)} className="group grid h-10 w-10 place-items-center rounded-lg transition hover:bg-white/10 disabled:cursor-wait" aria-label={`${score} out of 5`}>
                <Star size={22} className={`transition-colors group-hover:fill-[var(--accent)] group-hover:text-[var(--accent)] ${score <= draftScore ? "fill-[var(--accent)] text-[var(--accent)]" : "text-zinc-600"}`} />
              </button>
            ))}
          </div>
          {draftScore > 0 && (
            <div className="mt-3 border-t border-[var(--divider)] pt-3">
              <p className="mb-2 text-[10px] text-zinc-500">What influenced it? <span className={draftScore <= 2 ? "text-[#ff7777]" : "text-zinc-600"}>{draftScore <= 2 ? "Choose at least one" : "Optional"}</span></p>
              <div className="flex flex-wrap gap-1.5">
                {FEEDBACK_REASONS.map((reason) => {
                  const selected = draftReasons.has(reason.code);
                  const label = draftScore >= 4 ? reason.positive : reason.negative;
                  return (
                    <button key={reason.code} type="button" aria-pressed={selected} onClick={() => toggleReason(reason.code)} className={`rounded-[20px] px-[13px] py-[7px] text-[12px] font-semibold transition-colors ${selected ? "bg-[var(--accent-wash)] text-[var(--accent-text)]" : "bg-[var(--field)] text-[var(--text-2)] hover:bg-[var(--field-hover)]"}`}>
                      {label}
                    </button>
                  );
                })}
              </div>
              <button type="button" disabled={saving || lowScoreNeedsReason} onClick={() => onRate(draftScore, [...draftReasons])} title={lowScoreNeedsReason ? "Choose what needs improvement" : undefined} className="forge-btn-primary mt-3 w-full py-[10px] text-[12px]">
                {saving ? "Saving..." : "Save feedback"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const PROGRESS_STAGE_META = {
  prompt_generation: { title: "Composing the scene", code: "SCENE BRIEF" },
  prompt_ready: { title: "Scene direction locked", code: "BRIEF READY" },
  model_loading: { title: "Warming the render engine", code: "CHECKPOINT" },
  sampling: { title: "Resolving latent detail", code: "LATENT DENOISE" },
  refining: { title: "Polishing high-resolution detail", code: "DETAIL PASS" },
  finalizing: { title: "Converging the final pass", code: "FINAL PASS" },
  decoding: { title: "Developing the pixels", code: "VAE DECODE" },
  upscaling: { title: "Reconstructing fine detail", code: "PIXEL UPSCALE" },
  saving: { title: "Archiving your generation", code: "ARCHIVE" },
  queued: { title: "Opening the render session", code: "INITIALIZE" },
};

function useTweenedProgress(target) {
  const [displayValue, setDisplayValue] = useState(target);
  const valueRef = useRef(target);

  useEffect(() => {
    const from = valueRef.current;
    const distance = target - from;
    if (Math.abs(distance) < 0.1) return undefined;

    let frameId;
    let startedAt;
    const duration = Math.min(900, Math.max(420, Math.abs(distance) * 28));
    const draw = (now) => {
      startedAt ??= now;
      const elapsed = Math.min(1, (now - startedAt) / duration);
      const eased = 1 - ((1 - elapsed) ** 3);
      const next = from + (distance * eased);
      valueRef.current = next;
      setDisplayValue(next);
      if (elapsed < 1) frameId = requestAnimationFrame(draw);
    };
    frameId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frameId);
  }, [target]);

  return displayValue;
}

function GenerationProgress({ progress, promptEngine, resolutionMode }) {
  const percent = Math.max(0, Math.min(100, progress.percent ?? 0));
  const displayPercent = useTweenedProgress(percent);
  const stage = progress.stage ?? "queued";
  const stageMeta = PROGRESS_STAGE_META[stage] ?? PROGRESS_STAGE_META.queued;
  const engineName = promptEngine === "cloud" ? "Cloud API" : promptEngine === "ollama" ? "Ollama" : "Curated engine";
  const stageDescriptions = {
    queued: "Opening a generation slot and validating settings",
    prompt_generation: `Directing a new composition with ${engineName}`,
    prompt_ready: "Scene structure, exclusions, and model syntax validated",
    model_loading: "Loading the checkpoint and preparing GPU memory",
    sampling: progress.current_step && progress.total_steps
      ? `Denoising latent space - sample ${progress.current_step} of ${progress.total_steps}`
      : "Denoising latent space into a coherent composition",
    refining: progress.current_step && progress.total_steps
      ? `Reintroducing fine detail - sample ${progress.current_step} of ${progress.total_steps}`
      : "Reintroducing texture and high-resolution detail",
    finalizing: "Consolidating the final denoising pass",
    decoding: "Converting the latent result into full-color pixels",
    upscaling: resolutionMode === "super"
      ? "Reconstructing the final image at 2x output size"
      : "Reconstructing the final image at 1.5x output size",
    saving: "Writing the image, prompt, seed, and render metadata",
  };
  const stageDetail = progress.source === "estimate" && ["model_loading", "sampling", "refining"].includes(stage)
    ? `${stageDescriptions[stage]} - timed from your local render history`
    : stageDescriptions[stage] ?? "Advancing the render pipeline";
  const eta = progress.eta_seconds == null
    ? "ETA --"
    : progress.eta_seconds >= 60
      ? `~${Math.floor(progress.eta_seconds / 60)}m ${Math.ceil(progress.eta_seconds % 60)}s`
      : `~${Math.max(1, Math.ceil(progress.eta_seconds))}s`;
  const stages = [
    { key: "prompt", label: "Scene brief", matches: ["queued", "prompt_generation", "prompt_ready"] },
    { key: "model", label: "Checkpoint", matches: ["model_loading"] },
    { key: "sample", label: "Latent pass", matches: ["sampling"] },
    ...(resolutionMode !== "normal" ? [{ key: "refine", label: "Detail pass", matches: ["refining"] }] : []),
    { key: "decode", label: "Pixel decode", matches: ["finalizing", "decoding"] },
    ...(resolutionMode !== "normal" ? [{ key: "upscale", label: "Upscale", matches: ["upscaling", "final_upscale_4k", "final_upscale_8k", "upscale_1_8k", "upscale_1_12k", "upscale_2_12k"] }] : []),
    { key: "save", label: "Archive", matches: ["saving"] },
  ];
  let currentIndex = stages.findIndex((item) => item.matches.includes(stage));
  if (currentIndex < 0) currentIndex = 0;
  return (
    <div className="aiqg-generation-loader absolute inset-0 flex items-center justify-center overflow-hidden bg-[#090a0e]" aria-live="polite">
      <div className="aiqg-dot-field pointer-events-none absolute inset-0" aria-hidden="true" />
      <div className="relative z-10 w-[min(92%,900px)] px-4 text-center [text-shadow:0_2px_18px_#000]">
        <div className="inline-flex items-start justify-center whitespace-nowrap font-mono tabular-nums leading-[0.78]">
          <span className="aiqg-progress-number text-[clamp(7.5rem,12vw,12rem)] font-medium tracking-[-0.055em] text-white opacity-100">{Math.round(displayPercent)}</span>
          <span className="ml-3 mt-[clamp(.7rem,1.55vw,1.55rem)] text-[clamp(1.35rem,2.2vw,2.2rem)] font-semibold leading-none text-[var(--warn)] opacity-100">%</span>
        </div>
        <div className="mt-5 inline-flex items-center gap-2 font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--warn)]">
          <span className="aiqg-live-signal h-1.5 w-1.5 rounded-full bg-[var(--warn)]" />
          Live render session
        </div>
        <div key={stage} className="aiqg-stage-copy">
          <p className="mt-5 font-mono text-[10px] font-semibold uppercase tracking-[0.24em] text-[var(--warn)]">{stageMeta.code}</p>
          <h3 className="mt-2 text-[clamp(1.2rem,1.8vw,1.7rem)] font-semibold tracking-tight text-zinc-100">{stageMeta.title}</h3>
          <p className="mt-2 min-h-5 text-sm text-zinc-500">{stageDetail}</p>
        </div>
        <div className="mt-3 flex items-center justify-center gap-3 font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-600">
          <span>Pass {String(currentIndex + 1).padStart(2, "0")} / {String(stages.length).padStart(2, "0")}</span>
          <span className="h-px w-5 bg-[var(--divider)]" />
          <span>{eta}</span>
        </div>
        <div className="aiqg-progress-track mx-auto mt-6 h-px w-[min(100%,660px)] bg-white/[0.09]">
          <div className="aiqg-progress-fill relative h-full bg-[var(--warn)]" style={{ width: `${displayPercent}%` }} />
        </div>
        <ol className="aiqg-render-manifest mx-auto mt-6 flex w-[min(100%,820px)] flex-wrap items-center justify-center gap-x-3 gap-y-2" aria-label="Render pipeline">
          {stages.map((item, index) => {
            const complete = index < currentIndex;
            const active = index === currentIndex;
            return (
              <li key={item.key} className={`aiqg-render-stage inline-flex items-center gap-1.5 font-mono text-[9px] font-semibold uppercase tracking-[0.12em] transition-colors duration-300 ${active ? "is-active text-white" : complete ? "is-complete text-zinc-500" : "text-zinc-700"}`} aria-current={active ? "step" : undefined}>
                <span className="text-[8px] tabular-nums opacity-60">{String(index + 1).padStart(2, "0")}</span>
                <span>{item.label}</span>
                {complete && <Check size={10} strokeWidth={2.5} className="text-[var(--ok)]" />}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

function ProgressiveStageImage({ image, hidden, completed }) {
  const [loaded, setLoaded] = useState(false);

  useEffect(() => setLoaded(false), [image.id, hidden]);

  return (
    <span className="relative flex h-full w-full items-center justify-center overflow-hidden rounded-xl">
      <img
        src={image.thumbnail_url}
        alt=""
        aria-hidden="true"
        className={`absolute inset-0 h-full w-full scale-110 object-cover blur-2xl transition-opacity duration-300 ${hidden ? "opacity-45 brightness-50" : loaded ? "opacity-0" : "opacity-25"}`}
      />
      {!hidden && (
        <img
          src={image.full_url}
          alt={image.positive_prompt}
          decoding="async"
          onLoad={() => setLoaded(true)}
          className={`relative max-h-full max-w-full rounded-[12px] object-contain transition duration-300 ${loaded ? "opacity-100" : "opacity-0"} ${completed ? "aiqg-generation-complete" : ""}`}
        />
      )}
    </span>
  );
}

const PIXEL_UPSCALE_MULTIPLIERS = [1.5, 2, 3, 4];

function UpscaleMenu({ image, busy, isBusyOnThis, onDetailUpscale, onPixelUpscale }) {
  const itemClass = "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-xs font-medium text-zinc-300 outline-none transition data-[focus]:bg-zinc-700 data-[focus]:text-white disabled:cursor-not-allowed disabled:opacity-50";
  const alreadyDetailRendered = Boolean(image?.high_res || image?.super_res);
  return (
    <Menu>
      <MenuButton
        disabled={busy}
        className="flex h-9 items-center gap-2 rounded-[9px] bg-[rgba(18,16,26,0.92)] px-3 text-xs font-semibold text-[var(--text-ui)] backdrop-blur outline-none transition-colors hover:bg-[var(--field-hover)] disabled:cursor-not-allowed disabled:opacity-60 data-[open]:bg-[var(--accent-wash)] data-[open]:text-[var(--accent-text)]"
        aria-label="Upscale this image"
        title="Upscale this image"
      >
        {isBusyOnThis ? <LoaderCircle size={16} className="animate-spin" /> : <Maximize2 size={16} />}
        {isBusyOnThis ? "Upscaling…" : "Upscale"}
      </MenuButton>
      <MenuItems anchor="bottom end" transition className="z-[90] mt-2 w-72 origin-top-right rounded-[12px] bg-[var(--field)] p-2 shadow-[var(--shadow-menu)] transition duration-[120ms] [--anchor-gap:8px] focus:outline-none data-[closed]:-translate-y-[5px] data-[closed]:opacity-0">
        <div className="px-3 pb-2 pt-2">
          <p className="text-xs font-semibold text-white">Resize, no re-render</p>
          <p className="mt-0.5 text-[11px] text-zinc-500">Pure pixel upscale of this exact file. Works at any stage.</p>
        </div>
        <div className="grid grid-cols-4 gap-1.5 px-2 pb-2">
          {PIXEL_UPSCALE_MULTIPLIERS.map((multiplier) => (
            <MenuItem key={multiplier}>
              <button
                type="button"
                disabled={busy}
                onClick={() => onPixelUpscale(multiplier)}
                className="rounded-[8px] bg-[var(--raised)] py-2 text-center text-xs font-semibold text-[var(--text-ui)] outline-none transition-colors data-[focus]:bg-[var(--accent-wash)] data-[focus]:text-[var(--accent-text)] disabled:cursor-not-allowed disabled:opacity-50"
              >
                {multiplier}×
              </button>
            </MenuItem>
          ))}
        </div>
        <div className="my-1 h-px bg-white/5" />
        <div className="px-3 pb-1 pt-2">
          <p className="text-xs font-semibold text-white">Re-render with detail pass</p>
          <p className="mt-0.5 text-[11px] text-zinc-500">Same seed, redone with sharper hair, skin, and eye detail. Costs real GPU time.</p>
          {alreadyDetailRendered && (
            <p className="mt-2 rounded-[9px] bg-[var(--danger-wash)] px-2.5 py-2 text-[11px] leading-4 text-[#ff9d9d]">
              Already detail-rendered. Another latent pass can redraw composition and anatomy, so use Pixel Upscale instead.
            </p>
          )}
        </div>
        <MenuItem>
          <button type="button" disabled={busy} onClick={() => onDetailUpscale(false)} className={itemClass}>
            <Sparkles size={15} /> High detail pass
          </button>
        </MenuItem>
        <MenuItem>
          <button type="button" disabled={busy} onClick={() => onDetailUpscale(true)} className={itemClass}>
            <Sparkles size={15} /> Super detail pass
          </button>
        </MenuItem>
      </MenuItems>
    </Menu>
  );
}

export default function App() {
  const [form, setForm] = useState({ ui_mode: "simple", prompt: "", negative_prompt: "", prompt_run_id: null, model: "Forge default", style: "Photoreal", aspect_ratio: RATIOS[0], content_rating: "Safe", resolution_mode: "high", prompt_engine: "curated", creativity_level: "balanced", orientation: "mixed", cfg_scale: null, ollama_model: "", cloud_provider: "deepseek", cloud_model: "" });
  const [models, setModels] = useState(["Forge default"]);
  const [modelProfiles, setModelProfiles] = useState({});
  const [promptModels, setPromptModels] = useState([]);
  const [ollamaConnected, setOllamaConnected] = useState(false);
  const [cloudCredentials, setCloudCredentials] = useState(() => {
    try {
      const stored = JSON.parse(window.sessionStorage.getItem("aiqg_cloud_credentials") ?? "null");
      return stored && typeof stored === "object" ? { ...cloudProviderState(null), ...stored } : cloudProviderState(null);
    } catch {
      return cloudProviderState(null);
    }
  });
  const [account, setAccount] = useState(null);
  const [activeWorkspace, setActiveWorkspace] = useState(null);
  const [returningUser, setReturningUser] = useState(null);
  const [authenticatorAvailable, setAuthenticatorAvailable] = useState(false);
  const [authenticatorName, setAuthenticatorName] = useState("");
  const [authRequired, setAuthRequired] = useState(true);
  const [registrationOpen, setRegistrationOpen] = useState(false);
  const [authReady, setAuthReady] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const [workspaceSwitcherOpen, setWorkspaceSwitcherOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [queueSettingsOpen, setQueueSettingsOpen] = useState(false);
  const [backendSettingsOpen, setBackendSettingsOpen] = useState(false);
  const [authenticatorOpen, setAuthenticatorOpen] = useState(false);
  const [backends, setBackends] = useState([]);
  const [connected, setConnected] = useState(false);
  const [videoConnected, setVideoConnected] = useState(false);
  const [stalledJob, setStalledJob] = useState(null);
  const [images, setImages] = useState([]);
  const [historyCursor, setHistoryCursor] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [jobSubmitting, setJobSubmitting] = useState(false);
  const [promptPickerOpen, setPromptPickerOpen] = useState(false);
  const [surpriseLoading, setSurpriseLoading] = useState(false);
  const [surpriseNotice, setSurpriseNotice] = useState("");
  const [error, setError] = useState("");
  const [ratingImageId, setRatingImageId] = useState(null);
  const [ratingSaving, setRatingSaving] = useState(false);
  const [upscalingId, setUpscalingId] = useState(null);
  const [activeJob, setActiveJob] = useState(null);
  const [backgroundUpscaleJob, setBackgroundUpscaleJob] = useState(null);
  const [jobNotice, setJobNotice] = useState(null);
  const [knownJobs, setKnownJobs] = useState(() => {
    try {
      const stored = JSON.parse(window.sessionStorage.getItem("aiqg_job_handles") ?? "[]");
      return Array.isArray(stored) ? stored : [];
    } catch {
      return [];
    }
  });
  const [generationProgress, setGenerationProgress] = useState({ percent: 0, stage: "queued", source: "none", eta_seconds: null, current_image: null });
  const [completedImageId, setCompletedImageId] = useState(null);
  const imagesRef = useRef(images);
  const selectedImageIdRef = useRef(null);
  const submissionLockRef = useRef(false);
  imagesRef.current = images;
  selectedImageIdRef.current = (images[selectedIndex] ?? images[0] ?? null)?.id ?? null;
  const [routePath, setRoutePath] = useState(initialRoutePath);
  const page = pageForPath(routePath);
  const previousPageRef = useRef(page);
  const operations = useOperationsMonitor(account);

  const refreshBackends = useCallback(async () => {
    try {
      setBackends(await getBackends());
    } catch {
      // Preserve the last known states through a transient app/API failure.
    }
  }, []);

  useEffect(() => {
    refreshBackends();
    const interval = window.setInterval(refreshBackends, 10000);
    window.addEventListener("focus", refreshBackends);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", refreshBackends);
    };
  }, [refreshBackends]);

  useEffect(() => {
    const syncPage = () => setRoutePath(normalizeAppPath());
    window.addEventListener("popstate", syncPage);
    window.addEventListener("forge:navigate", syncPage);
    return () => {
      window.removeEventListener("popstate", syncPage);
      window.removeEventListener("forge:navigate", syncPage);
    };
  }, []);

  useEffect(() => {
    if (!loading || !activeJob) return undefined;
    let cancelled = false;
    const refresh = async () => {
      try {
        const job = await getJob(activeJob.job_id, activeJob.access_token);
        if (cancelled) return;
        setGenerationProgress((current) => {
          const next = {
            percent: job.progress_percent,
            stage: job.progress_stage,
            source: "job",
            eta_seconds: job.eta_seconds,
            current_image: job.current_image,
            current_step: job.current_step,
            total_steps: job.total_steps,
          };
          const unchanged = Object.keys(next).every((key) => Object.is(current[key], next[key]));
          return unchanged ? current : next;
        });
        if (job.status === "succeeded" && job.generation) {
          const generated = job.generation;
          setGenerationProgress((current) => ({ ...current, percent: 100, stage: "saving", eta_seconds: 0 }));
          setImages((current) => [generated, ...current.filter((item) => item.id !== generated.id)]);
          setSelectedIndex(0);
          setCompletedImageId(generated.id);
          setRatingImageId(null);
          if (activeJob.kind === "surprise_generate") {
            setForm((current) => ({ ...current, prompt: generated.positive_prompt, negative_prompt: generated.negative_prompt, prompt_run_id: null }));
            setSurpriseNotice(`Created with ${job.request_payload.prompt_engine ?? "curated"}.`);
          }
          setLoading(false);
          setActiveJob(null);
        } else if (["failed", "cancelled"].includes(job.status)) {
          setError(job.error_message ?? "The generation job failed.");
          setLoading(false);
          setActiveJob(null);
        }
      } catch (requestError) {
        if (!cancelled) setError(requestError.message);
      }
    };
    refresh();
    const interval = window.setInterval(refresh, 750);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [loading, activeJob]);

  useEffect(() => {
    if (!backgroundUpscaleJob) return undefined;
    let cancelled = false;
    const refresh = async () => {
      try {
        const job = await getJob(backgroundUpscaleJob.job_id, backgroundUpscaleJob.access_token);
        if (cancelled) return;
        if (job.status === "succeeded" && job.generation) {
          const generated = job.generation;
          const nextImages = [generated, ...imagesRef.current.filter((item) => item.id !== generated.id)];
          const preservedIndex = selectedImageIdRef.current == null
            ? 0
            : nextImages.findIndex((item) => item.id === selectedImageIdRef.current);
          setImages(nextImages);
          setSelectedIndex(preservedIndex >= 0 ? preservedIndex : 0);
          setCompletedImageId(generated.id);
          setUpscalingId(null);
          setBackgroundUpscaleJob(null);
          setJobNotice({ message: "Upscale complete", detail: "The new image is in Recent generations.", actionLabel: "View job" });
        } else if (["failed", "cancelled"].includes(job.status)) {
          setError(job.error_message ?? "The upscale job failed.");
          setUpscalingId(null);
          setBackgroundUpscaleJob(null);
          setJobNotice(null);
        }
      } catch (requestError) {
        if (!cancelled) setError(requestError.message);
      }
    };
    refresh();
    const interval = window.setInterval(refresh, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [backgroundUpscaleJob]);

  useEffect(() => {
    window.sessionStorage.setItem("aiqg_job_handles", JSON.stringify(knownJobs.slice(0, 100)));
  }, [knownJobs]);

  useEffect(() => {
    window.sessionStorage.setItem("aiqg_cloud_credentials", JSON.stringify(cloudCredentials));
  }, [cloudCredentials]);

  useEffect(() => {
    if (!completedImageId) return undefined;
    const timeout = window.setTimeout(() => setCompletedImageId(null), 1350);
    return () => window.clearTimeout(timeout);
  }, [completedImageId]);

  useEffect(() => {
    Promise.allSettled([getModels(), getPromptModels()]).then(([modelResult, promptModelResult]) => {
      if (modelResult.status === "fulfilled") {
        const available = modelResult.value.models;
        const profiles = modelResult.value.profiles ?? {};
        setModels(available);
        setModelProfiles(profiles);
        setConnected(modelResult.value.connected);
        setStalledJob(modelResult.value.stalled_job ?? null);
        setForm((current) => {
          const model = available.includes(current.model) ? current.model : available[0];
          const profile = profiles[model];
          const modelRatios = profile?.aspect_ratios?.length ? profile.aspect_ratios : RATIOS;
          return {
            ...current,
            model,
            style: profile?.styles.includes(current.style) ? current.style : (profile?.default_style ?? current.style),
            aspect_ratio: modelRatios.includes(current.aspect_ratio) ? current.aspect_ratio : modelRatios[0],
          };
        });
      }
      if (promptModelResult.status === "fulfilled") {
        const available = promptModelResult.value.models ?? [];
        setPromptModels(available);
        setOllamaConnected(Boolean(promptModelResult.value.connected && available.length));
        setForm((current) => ({
          ...current,
          ollama_model: available.includes(current.ollama_model)
            ? current.ollama_model
            : (promptModelResult.value.recommended ?? available[0] ?? ""),
        }));
      }
    });
  }, []);

  useEffect(() => {
    getAuthSession()
      .then(async (session) => {
        const user = session.authenticated ? session.user : null;
        setAuthenticatorAvailable(Boolean(session.authenticator_available));
        setAuthenticatorName(session.authenticator_display_name ?? user?.display_name ?? "");
        setRegistrationOpen(Boolean(session.registration_open));
        const unlockedThisBoot = Boolean(
          user && session.active_workspace && session.boot_id && window.localStorage.getItem(UNLOCKED_BOOT_KEY) === session.boot_id
        );
        if (unlockedThisBoot) {
          setAccount(user);
          setActiveWorkspace(session.active_workspace);
          setReturningUser(null);
          setAuthRequired(false);
          setAuthOpen(false);
          if (user.is_admin) {
            try {
              const history = await getHistory();
              setImages(history.items);
              setHistoryCursor(history.next_cursor);
              setSelectedIndex(0);
            } catch (historyError) {
              setError(historyError.message);
            }
          }
          return;
        }
        setAccount(null);
        setActiveWorkspace(null);
        setReturningUser(user);
        setAuthMode(session.authenticator_available ? "authenticator" : user ? "login" : session.registration_open ? "register" : "login");
        setAuthRequired(true);
        setAuthOpen(true);
        setImages([]);
        setHistoryCursor(null);
        setSelectedIndex(0);
      })
      .catch(() => {
        setAccount(null);
        setActiveWorkspace(null);
        setReturningUser(null);
        setAuthenticatorAvailable(false);
        setAuthenticatorName("");
        setAuthMode("login");
        setAuthRequired(true);
        setAuthOpen(true);
        setImages([]);
        setHistoryCursor(null);
      })
      .finally(() => setAuthReady(true));
  }, []);

  useEffect(() => {
    if (!authReady || authRequired || account?.is_admin || !["analytics", "gallery"].includes(page)) return;
    navigate(ROUTES.create, { replace: true });
  }, [account?.is_admin, authReady, authRequired, page]);

  useEffect(() => {
    const enteredGallery = previousPageRef.current !== "gallery" && page === "gallery";
    previousPageRef.current = page;
    if (!enteredGallery || authRequired || !account?.is_admin) return undefined;
    let active = true;
    setHistoryLoading(true);
    setError("");
    getHistory()
      .then((history) => {
        if (!active) return;
        setImages(history.items);
        setHistoryCursor(history.next_cursor);
        setSelectedIndex(0);
        setRatingImageId(null);
      })
      .catch((historyError) => {
        if (active) setError(historyError.message);
      })
      .finally(() => {
        if (active) setHistoryLoading(false);
      });
    return () => { active = false; };
  }, [account?.id, account?.is_admin, authRequired, page]);

  useEffect(() => {
    if (!authReady || authRequired || !activeWorkspace) return;
    if (activeWorkspace === "forgeai" && ["img", "vid", "bat"].includes(page)) {
      navigate(ROUTES.create, { replace: true });
    } else if (activeWorkspace === "forgeimg" && ["create", "vid", "bat"].includes(page)) {
      navigate(ROUTES.img, { replace: true });
    } else if (activeWorkspace === "forgevid" && ["create", "img", "bat"].includes(page)) {
      navigate(ROUTES.vid, { replace: true });
    } else if (activeWorkspace === "forgebat" && ["create", "img", "vid"].includes(page)) {
      navigate(ROUTES.bat, { replace: true });
    }
  }, [activeWorkspace, authReady, authRequired, page]);

  useEffect(() => {
    if (page !== "vid" || authRequired) return undefined;
    let active = true;
    const check = () => getVideoStatus().then((result) => { if (active) setVideoConnected(Boolean(result.connected)); }).catch(() => { if (active) setVideoConnected(false); });
    check();
    const timer = window.setInterval(check, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, [authRequired, page]);

  const toggleFavorite = async (image) => {
    const next = !image.favorite;
    setImages((current) => current.map((item) => item.id === image.id ? { ...item, favorite: next } : item));
    try {
      await setFavorite(image.id, next);
    } catch (requestError) {
      setImages((current) => current.map((item) => item.id === image.id ? { ...item, favorite: !next } : item));
      setError(requestError.message);
    }
  };
  const chooseThumbnail = useCallback((index) => {
    setSelectedIndex(index);
    setRatingImageId(null);
  }, []);
  const navigatePath = useCallback((path) => navigate(path), []);
  const navigatePage = useCallback((destination) => {
    const path = destination === "gallery"
      ? ROUTES.gallery
      : destination === "dashboard" || destination === "analytics"
        ? ROUTES.analytics
        : destination === "jobs" || destination === "deploy"
          ? ROUTES.deploy
          : ROUTES.create;
    navigate(path);
  }, []);

  const rememberJobHandle = useCallback((handle, kind) => {
    const tracked = { ...handle, kind };
    setKnownJobs((current) => [tracked, ...current.filter((item) => item.job_id !== handle.job_id)]);
    return tracked;
  }, []);

  const activateJob = useCallback((handle, kind, initialStage = "queued", background = false) => {
    const tracked = rememberJobHandle(handle, kind);
    if (background) {
      setBackgroundUpscaleJob(tracked);
      setJobNotice({ message: "Upscaling started", detail: "Your current image will stay here while Forge works.", actionLabel: "View progress" });
      return;
    }
    setActiveJob(tracked);
    setGenerationProgress({ percent: initialStage === "prompt_generation" ? 4 : 1, stage: initialStage, source: "job", eta_seconds: null, current_image: null });
    setLoading(true);
  }, [rememberJobHandle]);

  const openAuth = useCallback((mode = "login") => {
    setAuthMode(mode);
    setAuthOpen(true);
  }, []);

  const authenticate = async (mode, fields) => {
    const session = mode === "authenticator"
      ? await loginWithAuthenticator(fields.code)
      : mode === "register"
      ? await registerAccount(fields.displayName, fields.email, fields.password)
      : await loginAccount(returningUser?.email ?? fields.email, fields.password);
    // Credential validation is only the first half of unlocking the app.
    // Do not expose account-backed pages until the user also locks this
    // browser session to ForgeAI or ForgeIMG.
    setActiveWorkspace(null);
    setAuthenticatorAvailable(Boolean(session.authenticator_available));
    setAuthenticatorName(session.authenticator_display_name ?? session.user?.display_name ?? "");
    setRegistrationOpen(Boolean(session.registration_open));
    return session;
  };

  const continueAsGuest = useCallback(() => {
    setAccount(null);
    setActiveWorkspace(null);
    setReturningUser(null);
    setAuthRequired(false);
    setAuthOpen(false);
    setImages([]);
    setHistoryCursor(null);
    navigate(ROUTES.create, { replace: true });
  }, []);

  const chooseWorkspace = async (workspace, { announce = false } = {}) => {
    const session = await selectWorkspace(workspace);
    setAccount(session.user);
    setActiveWorkspace(session.active_workspace);
    setReturningUser(null);
    if (session.boot_id) window.localStorage.setItem(UNLOCKED_BOOT_KEY, session.boot_id);
    setAuthRequired(false);
    setAuthOpen(false);
    setWorkspaceSwitcherOpen(false);
    navigate(workspaceRoute(session.active_workspace), { replace: true });
    if (announce) {
      const continued = operations.summary.active;
      operations.pushToast({
        tone: "neutral",
        title: `Switched to ${WORKSPACES.find((item) => item.id === session.active_workspace)?.label ?? "workspace"}`,
        detail: continued ? `${continued} active operation${continued === 1 ? "" : "s"} continue in the background.` : "Your shared gallery, deployments, and analytics remain available.",
      });
    }
    if (session.user?.is_admin) {
      try {
        const history = await getHistory();
        setImages(history.items);
        setHistoryCursor(history.next_cursor);
        setSelectedIndex(0);
      } catch (historyError) {
        setError(historyError.message);
      }
    }
    return session;
  };

  const logout = async () => {
    try {
      const session = await logoutAccount();
      setRegistrationOpen(Boolean(session.registration_open));
      setAuthenticatorAvailable(Boolean(session.authenticator_available));
      setAuthenticatorName(session.authenticator_display_name ?? "");
      setAuthMode(session.authenticator_available ? "authenticator" : "login");
    } finally {
      window.localStorage.removeItem(UNLOCKED_BOOT_KEY);
      setAccount(null);
      setActiveWorkspace(null);
      setReturningUser(null);
      setAuthRequired(true);
      setAuthOpen(true);
      setImages([]);
      setHistoryCursor(null);
      setSelectedIndex(0);
      setRatingImageId(null);
      setSettingsOpen(false);
      setQueueSettingsOpen(false);
      setWorkspaceSwitcherOpen(false);
      setAuthenticatorOpen(false);
      setCloudCredentials(cloudProviderState(null));
      navigatePage("create");
      setForm((current) => ({
        ...current,
        prompt_engine: current.prompt_engine === "cloud" ? "curated" : current.prompt_engine,
        content_rating: "Safe",
        cloud_model: "",
        prompt_run_id: null,
      }));
    }
  };

  const saveCloudCredential = async (provider, apiKey) => {
    const validated = await testCloudCredential(provider, apiKey);
    const credential = {
      apiKey,
      models: validated.models,
      recommended: validated.recommended,
      selectedModel: validated.recommended,
    };
    setCloudCredentials((current) => ({ ...current, [provider]: credential }));
    setForm((current) => {
      const currentProviderConfigured = Boolean(cloudCredentials[current.cloud_provider]?.apiKey);
      if (currentProviderConfigured && current.cloud_provider !== provider) return current;
      return {
        ...current,
        cloud_provider: provider,
        cloud_model: validated.recommended,
        prompt_run_id: null,
      };
    });
    return validated;
  };

  const submit = async () => {
    if (submissionLockRef.current) return;
    setError("");
    if (!form.prompt.trim()) {
      setError("Enter a positive prompt before generating.");
      return;
    }
    if (!connected) {
      setError("Forge is offline. Launch Forge with the --api flag and try again.");
      return;
    }
    submissionLockRef.current = true;
    setJobSubmitting(true);
    try {
      const handle = await generateImage({ ...form, request_id: crypto.randomUUID() });
      activateJob(handle, "generate", "queued");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      submissionLockRef.current = false;
      setJobSubmitting(false);
    }
  };

  const requestAutomaticPrompt = () => getSurprisePrompt(
    form.model,
    form.style,
    form.content_rating,
    form.aspect_ratio,
    form.resolution_mode,
    form.prompt_engine,
    form.creativity_level,
    form.orientation,
    form.ollama_model,
    form.cloud_provider,
    form.cloud_model,
    cloudCredentials[form.cloud_provider]?.apiKey ?? "",
  );

  const surprise = async () => {
    setError("");
    setSurpriseNotice("");
    setSurpriseLoading(true);
    try {
      const prompts = await requestAutomaticPrompt();
      setForm((current) => ({
        ...current,
        prompt: prompts.prompt,
        negative_prompt: prompts.negative_prompt,
        prompt_run_id: prompts.prompt_run_id,
      }));
      setSurpriseNotice(prompts.notice ?? `Created with ${prompts.engine}.`);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSurpriseLoading(false);
    }
  };

  const simpleGenerate = async () => {
    if (submissionLockRef.current) return;
    setError("");
    setSurpriseNotice("");
    if (!connected) {
      setError("Forge is offline. Launch Forge with the --api flag and try again.");
      return;
    }
    if (form.prompt_engine === "reuse") {
      if (!form.prompt.trim()) {
        setError("Choose a past prompt first.");
        return;
      }
      await submit();
      return;
    }
    submissionLockRef.current = true;
    setJobSubmitting(true);
    try {
      const handle = await surpriseGenerateImage(
        form.model,
        form.style,
        form.content_rating,
        form.aspect_ratio,
        form.resolution_mode,
        form.prompt_engine,
        form.creativity_level,
        form.orientation,
        form.ollama_model,
        form.cloud_provider,
        form.cloud_model,
        cloudCredentials[form.cloud_provider]?.apiKey ?? "",
        crypto.randomUUID(),
        form.cfg_scale,
      );
      activateJob(handle, "surprise_generate", "prompt_generation");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      submissionLockRef.current = false;
      setJobSubmitting(false);
    }
  };

  const saveQueueLimit = async (maxActiveJobs) => {
    const updated = await updateAccountSettings(maxActiveJobs);
    setAccount(updated);
    return updated;
  };

  const submitTransform = async (payload) => {
    setError("");
    if (!connected) {
      setError("Forge is offline. Launch Forge with the --api flag and try again.");
      return;
    }
    setJobSubmitting(true);
    try {
      const handle = await transformImage(payload);
      activateJob(handle, payload.face_identity ? "character" : "img2img", "queued");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setJobSubmitting(false);
    }
  };

  const submitRating = async (generationId, score, reasons = []) => {
    setRatingSaving(true);
    setError("");
    try {
      const target = images.find((image) => image.id === generationId);
      const feedback = await rateImage(generationId, score, reasons, target?.access_token ?? null);
      setImages((current) => current.map((image) => image.id === generationId ? {
        ...image,
        user_rating: feedback.score,
        feedback_reasons: feedback.reasons,
      } : image));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setRatingSaving(false);
    }
  };

  const submitUpscale = async (generationId, superRes = false) => {
    setUpscalingId(generationId);
    setError("");
    try {
      const target = images.find((image) => image.id === generationId);
      const requestId = crypto.randomUUID();
      const handle = await upscaleImage(generationId, superRes, target?.access_token ?? null, requestId);
      activateJob(handle, "upscale", "queued", true);
    } catch (requestError) {
      setError(requestError.message);
      setUpscalingId(null);
    }
  };

  const submitPixelUpscale = async (generationId, multiplier) => {
    setUpscalingId(generationId);
    setError("");
    try {
      const target = images.find((image) => image.id === generationId);
      const handle = await pixelUpscaleImage(generationId, multiplier, target?.access_token ?? null);
      activateJob(handle, "pixel_upscale", "queued", true);
    } catch (requestError) {
      setError(requestError.message);
      setUpscalingId(null);
    }
  };

  const loadMoreHistory = async () => {
    if (!historyCursor || historyLoading) return;
    setHistoryLoading(true);
    setError("");
    try {
      const page = await getHistory(historyCursor);
      setImages((current) => {
        const known = new Set(current.map((image) => image.id));
        return [...current, ...page.items.filter((image) => !known.has(image.id))];
      });
      setHistoryCursor(page.next_cursor);
    } catch (historyError) {
      setError(historyError.message);
    } finally {
      setHistoryLoading(false);
    }
  };

  if (!authReady) {
    return (
      <main className="nyx-boot grid min-h-screen place-items-center px-6">
        <div className="flex flex-col items-center gap-[14px] text-center">
          <span className="nyx-brand-mark">N</span>
          <div className="flex flex-col gap-[6px]"><h1 className="text-[15px] font-bold uppercase tracking-[.16em] text-[var(--nyx-ink)]">NyxForge</h1><p className="forge-body">Checking your local session…</p></div>
        </div>
      </main>
    );
  }

  const chrome = chromeForPath(routePath);
  const statusLabel = page === "analytics"
    ? "Local telemetry"
    : page === "deploy"
      ? "Job orchestration"
      : page === "vid" ? videoConnected ? "ComfyUI connected" : "ComfyUI unavailable"
      : connected ? "Forge connected" : "Forge unavailable";

  return (
    <div className="min-h-screen bg-transparent pb-[34px] pt-[54px]">
      <AppTopBar
        product={chrome.product}
        breadcrumbs={chrome.breadcrumbs}
        connected={page === "analytics" ? true : page === "vid" ? videoConnected : connected}
        statusLabel={statusLabel}
        backends={backends}
        account={account}
        activeWorkspace={activeWorkspace}
        registrationOpen={authReady && registrationOpen}
        onNavigate={navigatePath}
        onOpenAuth={openAuth}
        onLogout={logout}
        onSwitchWorkspace={() => setWorkspaceSwitcherOpen(true)}
        onApiSettings={() => setSettingsOpen(true)}
        onQueueSettings={() => setQueueSettingsOpen(true)}
        onBackendSettings={() => setBackendSettingsOpen(true)}
        onAuthenticatorSettings={() => setAuthenticatorOpen(true)}
        currentPath={routePath}
        operationsSummary={operations.summary}
      />
      <PromptSourceDialog
        open={promptPickerOpen}
        onClose={() => setPromptPickerOpen(false)}
        images={images}
        model={form.model}
        hasMore={Boolean(historyCursor)}
        loadingMore={historyLoading}
        onLoadMore={loadMoreHistory}
        onUse={(image) => setForm((current) => ({
          ...current,
          prompt: image.positive_prompt ?? "",
          negative_prompt: image.negative_prompt ?? "",
          prompt_run_id: null,
          prompt_source_id: image.id,
        }))}
      />
      <AccountDialog open={authOpen || authRequired} initialMode={authMode} registrationOpen={registrationOpen} returningUser={returningUser} authenticatorAvailable={authenticatorAvailable} authenticatorName={authenticatorName} connected={connected} required={authRequired} onClose={() => setAuthOpen(false)} onAuthenticate={authenticate} onSelectWorkspace={chooseWorkspace} onContinueAsGuest={continueAsGuest} />
      <WorkspaceSwitcherDialog open={workspaceSwitcherOpen} activeWorkspace={activeWorkspace} operationsSummary={operations.summary} onClose={() => setWorkspaceSwitcherOpen(false)} onSelect={(workspace) => chooseWorkspace(workspace, { announce: true })} />
      <OperationToastViewport toasts={operations.toasts} onDismiss={operations.dismissToast} onNavigate={navigatePath} />
      <AuthenticatorSetupDialog open={authenticatorOpen} enabled={Boolean(account?.authenticator_enabled)} onClose={() => setAuthenticatorOpen(false)} onComplete={(updatedUser) => { setAccount(updatedUser); setAuthenticatorAvailable(true); setAuthenticatorName(updatedUser.display_name); }} />
      <CloudSettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} credentials={cloudCredentials} onSave={saveCloudCredential} />
      <QueueSettingsDialog open={queueSettingsOpen} account={account} onClose={() => setQueueSettingsOpen(false)} onSave={saveQueueLimit} />
      <BackendSettingsDialog open={backendSettingsOpen} backends={backends} onClose={() => setBackendSettingsOpen(false)} onRefresh={refreshBackends} onStatus={(next) => setBackends((current) => current.map((backend) => backend.id === next.id ? next : backend))} />
      {!authRequired && (page === "gallery" && account?.is_admin ? (
        <GalleryPage images={images} selectedIndex={selectedIndex} onChoose={chooseThumbnail} onToggleFavorite={toggleFavorite} hasMore={Boolean(historyCursor)} loadingMore={historyLoading} onLoadMore={loadMoreHistory} error={error} ratingImageId={ratingImageId} ratingSaving={ratingSaving} onOpenRating={setRatingImageId} onDismissRating={() => setRatingImageId(null)} onRate={submitRating} upscalingId={upscalingId} onUpscale={submitUpscale} onPixelUpscale={submitPixelUpscale} />
      ) : page === "img" ? (
        <ForgeIMGPage images={images} models={models} modelProfiles={modelProfiles} connected={connected} loading={jobSubmitting} error={error} onDismissError={() => setError("")} onSubmit={submitTransform} refreshSignal={knownJobs[0]?.job_id ?? 0} account={account} onNavigate={navigatePath} hasMore={Boolean(historyCursor)} loadingMore={historyLoading} onLoadMore={loadMoreHistory} promptModels={promptModels} ollamaConnected={ollamaConnected} cloudCredentials={cloudCredentials} promptSettings={form} onPromptSettingsChange={(updates) => setForm((current) => ({ ...current, ...updates }))} />
      ) : page === "vid" ? (
        <ForgeVIDPage images={images} hasMore={Boolean(historyCursor)} loadingMore={historyLoading} onLoadMore={loadMoreHistory} account={account} refreshSignal={knownJobs[0]?.job_id ?? 0} onNavigate={navigatePath} connected={videoConnected} promptModels={promptModels} cloudCredentials={cloudCredentials} promptSettings={form} />
      ) : page === "bat" ? (
        routePath === ROUTES.bat
          ? <ForgeBATHome onNavigate={navigatePath} />
          : <ForgeBATPage models={models} modelProfiles={modelProfiles} onNavigate={navigatePath} batchId={routePath.startsWith(`${ROUTES.bat}/batches/`) ? Number(routePath.split("/").at(-1)) : null} />
      ) : page === "analytics" && account?.is_admin ? (
        <Suspense fallback={<div className="grid min-h-[calc(100vh-60px)] place-items-center text-[13px] text-[var(--text-muted)]">Loading analytics…</div>}>
          <DashboardPage modelProfiles={modelProfiles} routePath={routePath} onNavigate={navigatePath} />
        </Suspense>
      ) : page === "deploy" ? (
        <Suspense fallback={<div className="grid min-h-[calc(100vh-60px)] place-items-center text-[13px] text-[var(--text-muted)]">Loading jobs…</div>}>
          <JobsPage routePath={routePath} onNavigate={navigatePath} account={account} guestHandles={knownJobs} onRememberJob={rememberJobHandle} />
        </Suspense>
      ) : (
        <main className="nyx-forgeai-workspace">
          <Sidebar form={form} setForm={setForm} onOpenPromptPicker={() => setPromptPickerOpen(true)} models={models} modelProfiles={modelProfiles} promptModels={promptModels} ollamaConnected={ollamaConnected} cloudCredentials={cloudCredentials} surpriseNotice={surpriseNotice} connected={connected} loading={jobSubmitting} surpriseLoading={surpriseLoading} onSurprise={surprise} onGenerate={submit} onSimpleGenerate={simpleGenerate} account={account} />
          <section className="nyx-jobs-stage">
            {stalledJob && <div className="mb-3 shrink-0 rounded-[9px] bg-[#f0ad4e]/12 px-[13px] py-[10px] text-[12px] font-medium leading-[1.45] text-[#f0c674]">Forge is still holding a finished “{stalledJob}” job, so new work will queue without starting. Restart reForge in Stability Matrix to clear it.</div>}
            <ErrorBanner error={error} onDismiss={() => setError("")} className="mb-3 shrink-0" />
            <ExecutionJobs account={account} refreshSignal={knownJobs[0]?.job_id ?? 0} onNavigate={navigatePath} modelProfiles={modelProfiles} jobKinds={["generate", "surprise_generate", "upscale", "pixel_upscale"]} description="Newest first" variant="forgeai" />
          </section>
        </main>
      ))}
      <AppStatusBar product={chrome.product} mode={chrome.product === "ForgeAI" ? form.ui_mode : null} activeWorkspace={activeWorkspace} operationsSummary={operations.summary} onSwitchWorkspace={() => setWorkspaceSwitcherOpen(true)} />
    </div>
  );
}
