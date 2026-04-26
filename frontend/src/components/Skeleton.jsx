export function SkeletonBlock({ className = "", style = null }) {
  return <span className={`skeleton-block ${className}`.trim()} style={style || undefined} aria-hidden="true" />;
}

export function SkeletonLines({ lines = 3, className = "" }) {
  return (
    <div className={`skeleton-lines ${className}`.trim()} aria-hidden="true">
      {Array.from({ length: lines }).map((_, index) => (
        <SkeletonBlock
          key={index}
          className={index === lines - 1 ? "skeleton-block--short" : ""}
        />
      ))}
    </div>
  );
}

export function SkeletonMetric({ className = "" }) {
  return (
    <div className={`skeleton-metric ${className}`.trim()} aria-hidden="true">
      <SkeletonBlock className="skeleton-block--label" />
      <SkeletonBlock className="skeleton-block--value" />
    </div>
  );
}
