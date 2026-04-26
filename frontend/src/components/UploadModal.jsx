import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { UploadCloud, X } from "lucide-react";
import { useVisionPlay } from "../context/VisionPlayContext";
import UploadForm from "./UploadForm";
import UploadProgressDialog from "./UploadProgressDialog";

export default function UploadModal({ closeTo }) {
  const navigate = useNavigate();
  const [submitError, setSubmitError] = useState("");
  const {
    createUpload,
    clearUploadError,
    uploadError,
    uploading,
    uploadProgress,
    uploadFilename
  } = useVisionPlay();

  useEffect(() => {
    function handleEscape(event) {
      if (!uploading && event.key === "Escape") {
        navigate(closeTo, { replace: true });
      }
    }

    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("keydown", handleEscape);
      clearUploadError();
    };
  }, [closeTo, navigate, clearUploadError, uploading]);

  async function handleSubmit(file) {
    setSubmitError("");

    try {
      const nextVideoId = await createUpload(file);
      navigate(`/workspace/results/${nextVideoId}`, { replace: true });
    } catch (requestError) {
      setSubmitError(requestError.message);
    }
  }

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onClick={(event) => {
        if (!uploading && event.target === event.currentTarget) {
          navigate(closeTo, { replace: true });
        }
      }}
    >
      <div className="modal-card" role="dialog" aria-modal="true" aria-labelledby="upload-modal-title" data-aos="zoom-in">
        <div className="modal-card__header">
          <div className="modal-card__copy">
            <span className="section-kicker">Upload</span>
            <h2 id="upload-modal-title">Upload video</h2>
            <p>Choose a file to start a new local analysis.</p>
          </div>

          {uploading ? (
            <button className="modal-close" type="button" disabled>
              <UploadCloud size={16} strokeWidth={2.1} aria-hidden="true" />
              Uploading
            </button>
          ) : (
            <Link className="modal-close" to={closeTo} replace>
              <X size={16} strokeWidth={2.1} aria-hidden="true" />
              Close
            </Link>
          )}
        </div>

        <div className="modal-card__body">
          <UploadForm
            onSubmit={handleSubmit}
            uploading={uploading}
            uploadProgress={uploadProgress}
            error={submitError || uploadError}
          />
        </div>

        {uploading ? <UploadProgressDialog filename={uploadFilename} progress={uploadProgress} /> : null}
      </div>
    </div>
  );
}
