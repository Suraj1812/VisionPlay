import { LoaderCircle } from "lucide-react";

export default function UploadProgressDialog({ filename, progress }) {
  const clampedProgress = Math.min(100, Math.max(0, Math.round(progress || 0)));
  const isFinishing = clampedProgress >= 100;

  return (
    <div className="upload-progress-dialog" role="status" aria-live="polite" aria-atomic="true">
      <div className="upload-progress-dialog__card">
        <span className="section-kicker">{isFinishing ? "Finalizing" : "Uploading"}</span>
        <strong>
          <LoaderCircle size={18} strokeWidth={2.1} aria-hidden="true" />
          {isFinishing ? "100%" : `${clampedProgress}%`}
        </strong>
        <p>{filename || "Preparing your video upload."}</p>

        <div
          className="upload-progress-dialog__track"
          aria-label={`Upload progress ${clampedProgress} percent`}
          aria-valuemax={100}
          aria-valuemin={0}
          aria-valuenow={clampedProgress}
          role="progressbar"
        >
          <span className="upload-progress-dialog__bar" style={{ width: `${clampedProgress}%` }} />
        </div>

        <small>
          {isFinishing ? "Preparing your new session..." : "Keep this window open while the upload finishes."}
        </small>
      </div>
    </div>
  );
}
