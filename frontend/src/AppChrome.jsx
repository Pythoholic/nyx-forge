import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { useEffect, useState } from "react";
import {
  ArrowRightLeft,
  BarChart3,
  BriefcaseBusiness,
  CircleUserRound,
  Clapperboard,
  ImageUp,
  Images,
  KeyRound,
  Layers,
  LoaderCircle,
  LogIn,
  LogOut,
  Palette,
  ServerCog,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  UserPlus,
} from "lucide-react";
import { applyNyxTheme, NYX_THEMES, readNyxTheme } from "./nyxTheme";

export const ROUTES = Object.freeze({
  create: "/forge-ai",
  gallery: "/forge-ai/gallery",
  img: "/forge-img",
  vid: "/forge-vid",
  bat: "/forge-bat",
  deploy: "/forge-deploy",
  analytics: "/forge-analytics",
});

export const PRODUCTS = Object.freeze({
  ForgeAI: { tagline: "Create", icon: Sparkles },
  ForgeIMG: { tagline: "Transform", icon: ImageUp },
  ForgeVID: { tagline: "Motion", icon: Clapperboard },
  ForgeBAT: { tagline: "Batch studio", icon: Layers },
  ForgeDeploy: { tagline: "Operations", icon: BriefcaseBusiness },
  ForgeAnalytics: { tagline: "Telemetry", icon: BarChart3 },
});

export function normalizeAppPath(pathname = window.location.pathname) {
  const path = pathname.replace(/\/+$/, "") || "/";
  if (path === "/" || path === "/index.html") return ROUTES.create;
  return path;
}

export function navigate(path, { replace = false } = {}) {
  const target = normalizeAppPath(path);
  window.history[replace ? "replaceState" : "pushState"]({}, "", target);
  window.dispatchEvent(new CustomEvent("forge:navigate", { detail: { path: target } }));
}

function RouteLink({ to, onNavigate, className, children, ...props }) {
  return <a href={to} className={className} onClick={(event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    onNavigate(to);
  }} {...props}>{children}</a>;
}

export function NyxThemePicker({ showLabel = false }) {
  const [theme, setTheme] = useState(() => readNyxTheme());
  useEffect(() => { applyNyxTheme(theme); }, [theme]);
  return <div className="nyx-theme-picker" aria-label="NyxForge accent theme">
    {NYX_THEMES.map((item) => <button key={item.id} type="button" aria-label={`Use ${item.label} theme`} aria-pressed={theme === item.id} title={item.label} style={{ "--swatch": item.color }} onClick={() => setTheme(item.id)} />)}
    {showLabel && <span className="nyx-theme-name">{NYX_THEMES.find((item) => item.id === theme)?.label ?? "Solar"}</span>}
  </div>;
}

export function AccountMenu({ user, activeWorkspace, registrationOpen, onOpenAuth, onLogout, onNavigate, onSwitchWorkspace, onApiSettings, onQueueSettings, onBackendSettings, onAuthenticatorSettings }) {
  const nameParts = user?.display_name?.trim().split(/\s+/) ?? [];
  const initials = nameParts.length > 1
    ? nameParts.slice(0, 2).map((part) => part[0]).join("").toUpperCase()
    : (nameParts[0]?.slice(0, 2).toUpperCase() ?? "");
  const itemClass = "nyx-menu-item";
  const routeItems = [
    ...(activeWorkspace === "forgeai" ? [[ROUTES.create, Palette, "ForgeAI"]] : []),
    ...(user?.is_admin ? [[ROUTES.gallery, Images, "Gallery"]] : []),
    ...(activeWorkspace === "forgeimg" ? [[ROUTES.img, ImageUp, "ForgeIMG"]] : []),
    ...(activeWorkspace === "forgevid" ? [[ROUTES.vid, Clapperboard, "ForgeVID"]] : []),
    ...(activeWorkspace === "forgebat" ? [[ROUTES.bat, Layers, "ForgeBAT"]] : []),
    [ROUTES.deploy, BriefcaseBusiness, "ForgeDeploy"],
    ...(user?.is_admin ? [[ROUTES.analytics, BarChart3, "ForgeAnalytics"]] : []),
  ];
  return <Menu>
    <MenuButton className="nyx-account-button" aria-label="Open account menu" title={user ? user.display_name : "Account menu"}>{initials || <CircleUserRound size={15} />}</MenuButton>
    <MenuItems anchor="bottom end" transition className="nyx-menu [--anchor-gap:8px] data-[closed]:-translate-y-1 data-[closed]:opacity-0">
      <div className="nyx-menu-account"><strong>{user?.display_name ?? "Guest session"}</strong><span>{user?.email ?? "Sign in for history and settings"}</span></div>
      <div className="nyx-menu-rule" />
      {!user && <>
        <MenuItem><button type="button" onClick={() => onOpenAuth("login")} className={itemClass}><LogIn size={14} /> Admin sign in</button></MenuItem>
        {registrationOpen && <MenuItem><button type="button" onClick={() => onOpenAuth("register")} className={itemClass}><UserPlus size={14} /> Create admin account</button></MenuItem>}
      </>}
      {routeItems.map(([path, Icon, label]) => <MenuItem key={path}><button type="button" onClick={() => onNavigate(path)} className={itemClass}><Icon size={14} /> {label}</button></MenuItem>)}
      {user && activeWorkspace && <MenuItem><button type="button" onClick={onSwitchWorkspace} className={itemClass}><ArrowRightLeft size={14} /> Switch workspace</button></MenuItem>}
      {user?.is_admin && <MenuItem><button type="button" onClick={onApiSettings} className={itemClass}><KeyRound size={14} /> API configuration</button></MenuItem>}
      {user?.is_admin && <MenuItem><button type="button" onClick={onBackendSettings} className={itemClass}><ServerCog size={14} /> Forge backends</button></MenuItem>}
      {user && <MenuItem><button type="button" onClick={onQueueSettings} className={itemClass}><SlidersHorizontal size={14} /> Settings</button></MenuItem>}
      {user?.is_admin && <MenuItem><button type="button" onClick={onAuthenticatorSettings} className={itemClass}><ShieldCheck size={14} /> Authenticator sign-in</button></MenuItem>}
      {user && <><div className="nyx-menu-rule" /><MenuItem><button type="button" onClick={onLogout} className={`${itemClass} is-danger`}><LogOut size={14} /> Sign out</button></MenuItem></>}
    </MenuItems>
  </Menu>;
}

export function AppTopBar({ product, breadcrumbs = [], connected, statusLabel, backends = [], account, activeWorkspace, registrationOpen, onNavigate, onOpenAuth, onLogout, onSwitchWorkspace, onApiSettings, onQueueSettings, onBackendSettings, onAuthenticatorSettings, currentPath, operationsSummary }) {
  const navItems = [
    ...(activeWorkspace === "forgeai" ? [[ROUTES.create, "ForgeAI"]] : []),
    ...(account?.is_admin ? [[ROUTES.gallery, "Gallery"]] : []),
    ...(activeWorkspace === "forgeimg" ? [[ROUTES.img, "ForgeIMG"]] : []),
    ...(activeWorkspace === "forgevid" ? [[ROUTES.vid, "ForgeVID"]] : []),
    ...(activeWorkspace === "forgebat" ? [[ROUTES.bat, "ForgeBAT"]] : []),
    [ROUTES.deploy, "ForgeDeploy"],
    ...(account?.is_admin ? [[ROUTES.analytics, "ForgeAnalytics"]] : []),
  ];
  const home = product === "ForgeAI" ? ROUTES.create : product === "ForgeIMG" ? ROUTES.img : product === "ForgeVID" ? ROUTES.vid : product === "ForgeBAT" ? ROUTES.bat : product === "ForgeDeploy" ? ROUTES.deploy : ROUTES.analytics;
  const live = connected !== false;
  return <header className="nyx-topbar">
    <div className="nyx-topbar-left">
      <RouteLink to={home} onNavigate={onNavigate} className="nyx-brand"><span className="nyx-brand-mark">N</span><strong>{product.toUpperCase()}</strong><span className="nyx-product-badge">{PRODUCTS[product]?.tagline ?? "Workspace"}</span></RouteLink>
      {backends.length ? <div className="nyx-backends" aria-label="Model runner status">{backends.map((backend) => <span key={backend.id} className={backend.reachable ? "is-online" : backend.running ? "is-starting" : "is-offline"} title={backend.loaded_checkpoint ?? backend.error ?? backend.label}><i />{backend.label}<b>{backend.reachable ? "online" : backend.running ? "starting" : "offline"}</b></span>)}</div> : <span className={`nyx-connection ${live ? "is-online" : "is-offline"}`}><i />{statusLabel ?? (live ? "Connected" : "Unavailable")}</span>}
      <nav className="nyx-product-nav" aria-label="Products">{navItems.map(([path, label]) => {
        const active = label === "Gallery" ? currentPath === ROUTES.gallery : product === label && currentPath !== ROUTES.gallery;
        return <RouteLink key={path} to={path} onNavigate={onNavigate} aria-current={active ? "page" : undefined} className={active ? "is-active" : ""}>{label}</RouteLink>;
      })}</nav>
      {breadcrumbs.length > 1 && <span className="nyx-breadcrumb">/ {breadcrumbs.at(-1)?.label}</span>}
    </div>
    <div className="nyx-topbar-right">
      {operationsSummary?.active > 0 && <button type="button" onClick={() => onNavigate(ROUTES.deploy)} className="nyx-active-jobs"><LoaderCircle size={12} className="animate-spin" />{operationsSummary.active} active</button>}
      {backends.length > 0 && <span className={`nyx-suite-status ${backends.every((backend) => backend.reachable) ? "is-online" : "is-degraded"}`}><i />{backends.every((backend) => backend.reachable) ? "All systems connected" : `${backends.filter((backend) => !backend.reachable).length} system unavailable`}</span>}
      <NyxThemePicker />
      <AccountMenu user={account} activeWorkspace={activeWorkspace} registrationOpen={registrationOpen} onOpenAuth={onOpenAuth} onLogout={onLogout} onNavigate={onNavigate} onSwitchWorkspace={onSwitchWorkspace} onApiSettings={onApiSettings} onQueueSettings={onQueueSettings} onBackendSettings={onBackendSettings} onAuthenticatorSettings={onAuthenticatorSettings} />
    </div>
  </header>;
}

export function AppStatusBar({ product, mode, activeWorkspace, operationsSummary, onSwitchWorkspace }) {
  const [theme, setTheme] = useState(() => readNyxTheme());
  useEffect(() => {
    const listener = (event) => setTheme(event.detail.theme);
    window.addEventListener("nyxforge:theme", listener);
    return () => window.removeEventListener("nyxforge:theme", listener);
  }, []);
  return <footer className="nyx-statusbar">
    <span className="nyx-status-mode">&gt;_ {product}</span>{mode && <span><b>mode:</b> {mode}</span>}<span><b>workspace:</b> {activeWorkspace ?? "guest"}</span><span><b>theme:</b> {theme}</span>
    {activeWorkspace && <button type="button" onClick={onSwitchWorkspace}>&lt; switch workspace</button>}
    <span className="nyx-status-queue"><b>queue:</b> {operationsSummary?.active ?? 0} active</span>
  </footer>;
}
