export function stripUploadSuffix(pathname) {
  if (!pathname.endsWith("/upload")) {
    return pathname;
  }

  const nextPath = pathname.replace(/\/upload$/, "");
  return nextPath || "/workspace";
}

export function buildUploadRoute(pathname) {
  if (!pathname.startsWith("/workspace")) {
    return "/workspace/upload";
  }

  const basePath = stripUploadSuffix(pathname);
  return `${basePath}/upload`;
}
