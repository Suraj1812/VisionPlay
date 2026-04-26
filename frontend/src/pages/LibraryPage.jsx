import { useState } from "react";
import { ChevronDown, Search, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";
import WorkspaceOptionsPanel from "../components/WorkspaceOptionsPanel";
import SessionList from "../components/SessionList";
import { SkeletonBlock } from "../components/Skeleton";
import { useVisionPlay } from "../context/VisionPlayContext";

const FILTERS = [
  { value: "all", label: "All" },
  { value: "processing", label: "Processing" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" }
];

export default function LibraryPage() {
  const { isBooting, sessions } = useVisionPlay();
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  const normalizedQuery = query.trim().toLowerCase();
  const filteredSessions = sessions.filter((session) => {
    const matchesStatus = statusFilter === "all" ? true : session.status === statusFilter;
    const matchesQuery = normalizedQuery
      ? `${session.filename || ""} ${session.videoId}`.toLowerCase().includes(normalizedQuery)
      : true;

    return matchesStatus && matchesQuery;
  });

  return (
    <div className="page-stack">
      <section className="library-shell" data-aos="fade-up">
        <div className="library-shell__main">
          <SessionList
            sessions={filteredSessions}
            title="Sessions"
            loading={isBooting}
            actions={
              <div className="library-inline-toolbar">
                {isBooting ? (
                  <>
                    <SkeletonBlock className="skeleton-block--input library-inline-toolbar__search-skeleton" />
                    <SkeletonBlock className="skeleton-block--chip library-inline-toolbar__select-skeleton" />
                  </>
                ) : (
                  <>
                    <div className="search-field search-field--compact">
                      <div className="search-field__control">
                        <Search
                          size={16}
                          strokeWidth={2.1}
                          aria-hidden="true"
                          className="search-field__icon"
                        />
                        <input
                          type="search"
                          value={query}
                          onChange={(event) => setQuery(event.target.value)}
                          placeholder="Search by filename or ID"
                          aria-label="Search sessions"
                        />
                      </div>
                    </div>

                    <label className="library-select">
                      <select
                        value={statusFilter}
                        onChange={(event) => setStatusFilter(event.target.value)}
                        aria-label="Filter sessions by status"
                      >
                        {FILTERS.map((filter) => (
                          <option key={filter.value} value={filter.value}>
                            {filter.label}
                          </option>
                        ))}
                      </select>
                      <ChevronDown
                        size={16}
                        strokeWidth={2.1}
                        aria-hidden="true"
                        className="library-select__icon"
                      />
                    </label>
                  </>
                )}
              </div>
            }
            emptyTitle={query || statusFilter !== "all" ? "No matching sessions" : "No saved sessions yet"}
            emptyCopy={
              query || statusFilter !== "all"
                ? "Try another search."
                : "Completed uploads appear here."
            }
            emptyAction={
              query || statusFilter !== "all" ? null : (
                <Link className="secondary-button secondary-button--small secondary-button--icon" to="/workspace/upload">
                  <UploadCloud size={16} strokeWidth={2.1} aria-hidden="true" />
                  Upload video
                </Link>
              )
            }
          />
        </div>

        <aside className="library-shell__side">
          <WorkspaceOptionsPanel loading={isBooting} />
        </aside>
      </section>
    </div>
  );
}
