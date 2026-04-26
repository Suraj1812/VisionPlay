import { Link } from "react-router-dom";
import WorkspaceOptionsPanel from "../components/WorkspaceOptionsPanel";
import ProcessingStatus from "../components/ProcessingStatus";
import SessionList from "../components/SessionList";
import { useVisionPlay } from "../context/VisionPlayContext";

export default function WorkspacePage() {
  const { isBooting, latestActiveSession, latestCompletedSession, sessions } = useVisionPlay();
  const activeSession = latestActiveSession || latestCompletedSession;

  return (
    <div className="page-stack page-stack--workspace">
      <div className="workspace-overview-grid">
        <div className="workspace-overview-grid__main">
          <ProcessingStatus session={activeSession} loading={isBooting} />
        </div>

        <aside className="workspace-overview-grid__side">
          <WorkspaceOptionsPanel loading={isBooting} />
        </aside>
      </div>

      <div data-aos="fade-up" data-aos-delay="60">
        <SessionList
          sessions={sessions.slice(0, 5)}
          title="Recent sessions"
          emptyTitle="No sessions yet"
          emptyCopy="Uploads appear here."
          loading={isBooting}
          actions={
            sessions.length ? (
              <Link className="secondary-button secondary-button--small" to="/workspace/library">
                View all
              </Link>
            ) : null
          }
        />
      </div>
    </div>
  );
}
