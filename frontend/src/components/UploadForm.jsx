import { useRef, useState } from "react";
import { ArrowUpRight, FolderUp, ShieldCheck, Video } from "lucide-react";

function formatFileSize(file) {
  if (!file) {
    return "No file selected";
  }

  const sizeMb = file.size / 1024 / 1024;
  if (sizeMb < 1) {
    return `${Math.max(1, Math.round(file.size / 1024))} KB`;
  }

  return `${sizeMb.toFixed(sizeMb >= 10 ? 0 : 1)} MB`;
}

export default function UploadForm({ onSubmit, uploading, uploadProgress, error }) {
  const [file, setFile] = useState(null);
  const [localError, setLocalError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  function updateFile(nextFile) {
    setLocalError("");
    setFile(nextFile || null);
  }

  function handleFileChange(event) {
    updateFile(event.target.files?.[0] || null);
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      setLocalError("Select a video file.");
      return;
    }

    await onSubmit(file);
  }

  const visibleError = localError || error;

  return (
    <form className="upload-form" onSubmit={handleSubmit}>
      <input
        ref={inputRef}
        className="upload-form__input"
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska"
        onChange={handleFileChange}
        disabled={uploading}
      />

      <div
        className={isDragging ? "upload-dropzone upload-dropzone--dragging" : "upload-dropzone"}
        role="button"
        tabIndex={uploading ? -1 : 0}
        onClick={() => {
          if (!uploading) {
            inputRef.current?.click();
          }
        }}
        onKeyDown={(event) => {
          if (uploading) {
            return;
          }
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (!uploading) {
            setIsDragging(true);
          }
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (!uploading) {
            updateFile(event.dataTransfer.files?.[0] || null);
          }
        }}
      >
        <span className="section-kicker">Drag and drop</span>
        <strong>{file ? file.name : "Drop a video here"}</strong>
        <p>Supports MP4, MOV, AVI, WebM, and MKV files.</p>
        <span className="upload-dropzone__hint">
          <ShieldCheck size={15} strokeWidth={2.1} aria-hidden="true" />
          Everything stays on this device unless you move the files yourself.
        </span>

        <button type="button" className="secondary-button secondary-button--small secondary-button--icon upload-dropzone__action">
          <FolderUp size={15} strokeWidth={2.1} aria-hidden="true" />
          {file ? "Replace file" : "Browse files"}
        </button>
      </div>

      {visibleError ? <div className="error-banner">{visibleError}</div> : null}

      <div className="upload-form__footer">
        <div className="upload-file-card">
          <span className="metric-label">File</span>
          <strong title={file?.name || "No file selected"}>
            <Video size={16} strokeWidth={2.1} aria-hidden="true" />
            {file?.name || "No file selected"}
          </strong>
          <small>{file ? formatFileSize(file) : "Choose a video to begin."}</small>
        </div>

        <button className="primary-button primary-button--icon" type="submit" disabled={uploading}>
          <ArrowUpRight size={16} strokeWidth={2.1} aria-hidden="true" />
          {uploading ? `Uploading ${Math.round(uploadProgress || 0)}%` : "Start"}
        </button>
      </div>
    </form>
  );
}
