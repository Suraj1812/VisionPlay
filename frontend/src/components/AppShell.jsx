import { useEffect } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  Activity,
  Camera,
  CheckCircle2,
  Library,
  LayoutDashboard,
  Sparkles,
  UploadCloud
} from "lucide-react";
import { useVisionPlay } from "../context/VisionPlayContext";
import { buildUploadRoute, stripUploadSuffix } from "../utils/routes";
import UploadModal from "./UploadModal";
import { SkeletonBlock } from "./Skeleton";

function ShellLink({ to, icon: Icon, children }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        isActive ? "workspace-nav__link workspace-nav__link--active" : "workspace-nav__link"
      }
      end={to === "/workspace"}
    >
      {Icon ? <Icon size={16} strokeWidth={2} aria-hidden="true" /> : null}
      {children}
    </NavLink>
  );
}

export default function AppShell() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isBooting, latestActiveSession, sessions } = useVisionPlay();
  const uploadTarget = buildUploadRoute(location.pathname);
  const isUploadRoute = location.pathname.endsWith("/upload");
  const modalCloseTarget = stripUploadSuffix(location.pathname);
  const completedCount = sessions.filter((session) => session.status === "completed").length;
  const activeCount = sessions.filter(
    (session) => session.status === "pending" || session.status === "processing"
  ).length;

  useEffect(() => {
    function handleShortcut(event) {
      const target = event.target;
      const isTypingTarget =
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT");

      if (isTypingTarget || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) {
        return;
      }

      if (event.key.toLowerCase() === "u") {
        event.preventDefault();
        navigate(uploadTarget);
      }
    }

    window.addEventListener("keydown", handleShortcut);
    return () => {
      window.removeEventListener("keydown", handleShortcut);
    };
  }, [navigate, uploadTarget]);

  return (
    <div className="workspace-shell">
      <header className="workspace-header" data-aos="fade-down">
        <div className="workspace-header__row">
          <Link className="workspace-brand" to="/workspace">
            <img className="workspace-brand__logo" src="/visionplay-logo.svg" alt="" />
            <span className="workspace-brand__copy">
              <strong>VisionPlay</strong>
            </span>
          </Link>

          <nav className="workspace-nav" aria-label="Workspace">
            <ShellLink to="/workspace" icon={LayoutDashboard}>Overview</ShellLink>
            <ShellLink to="/workspace/live" icon={Camera}>Live Camera</ShellLink>
            <ShellLink to="/workspace/library" icon={Library}>Library</ShellLink>
          </nav>

          <div className="workspace-header__actions">
            {isBooting ? (
              <div className="workspace-header__loading">
                <SkeletonBlock className="skeleton-block--chip" />
                <SkeletonBlock className="skeleton-block--button" />
              </div>
            ) : (
              <>
                <div className="workspace-header__stats" aria-label="Workspace status">
                  <span className="workspace-stat">
                    <CheckCircle2 size={15} strokeWidth={2.1} aria-hidden="true" />
                    {completedCount} ready
                  </span>
                  <span className="workspace-stat">
                    <Activity size={15} strokeWidth={2.1} aria-hidden="true" />
                    {activeCount} active
                  </span>
                  {latestActiveSession ? (
                    <span className="workspace-stat workspace-stat--accent">
                      <Sparkles size={15} strokeWidth={2.1} aria-hidden="true" />
                      {latestActiveSession.processingProgress || 0}% latest
                    </span>
                  ) : null}
                </div>

                <Link className="primary-button primary-button--icon" to={uploadTarget}>
                  <UploadCloud size={17} strokeWidth={2.1} aria-hidden="true" />
                  Upload
                </Link>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="workspace-main">
        <Outlet />
      </main>

      {isUploadRoute ? <UploadModal closeTo={modalCloseTarget} /> : null}
    </div>
  );
}
