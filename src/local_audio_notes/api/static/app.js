// Minimal, same-origin only. No third-party code, no inline handlers (CSP forbids them).
"use strict";

function seekLinks() {
  const player = document.getElementById("player");
  if (!player) return;
  document.querySelectorAll("a.seek[data-start]").forEach((link) => {
    link.addEventListener("click", () => {
      const start = Number(link.dataset.start);
      if (Number.isFinite(start)) {
        player.currentTime = start;
        player.play().catch(() => {});
      }
    });
  });
}

async function upload(file, kind, form, onProgress) {
  const limit = Number(form.dataset[kind === "audio" ? "audioLimit" : "transcriptLimit"]);
  if (file.size === 0 || file.size > limit) throw new Error(`${kind} file is empty or too large`);
  // XMLHttpRequest for upload progress; the body is the raw file, never a filename.
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `/api/upload?kind=${kind}`);
    request.setRequestHeader("X-CSRF-Token", form.dataset.csrf);
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    request.addEventListener("load", () => {
      if (request.status === 200) resolve(JSON.parse(request.responseText).upload_id);
      else reject(new Error(`${kind} upload was refused (HTTP ${request.status})`));
    });
    request.addEventListener("error", () => reject(new Error(`${kind} upload failed`)));
    request.send(file);
  });
}

function importForm() {
  const form = document.getElementById("import-form");
  if (!form) return;
  const status = document.getElementById("upload-status");
  const progress = document.getElementById("upload-progress");
  form.addEventListener("submit", async (event) => {
    if (form.elements.audio_upload.value) return; // uploads done; let the form submit
    event.preventDefault();
    const audio = document.getElementById("audio-file").files[0];
    const transcript = document.getElementById("transcript-file").files[0];
    if (!audio) return;
    form.querySelector("button[type=submit]").disabled = true;
    progress.hidden = false;
    try {
      status.textContent = "Uploading audio…";
      form.elements.audio_upload.value = await upload(audio, "audio", form, (p) => {
        progress.value = Math.round(p * 100);
      });
      if (transcript) {
        const suffix = (transcript.name.match(/\.(txt|srt|json)$/i) || [""])[0].toLowerCase();
        if (!suffix) throw new Error("Transcript must be .txt, .srt or .json");
        status.textContent = "Uploading transcript…";
        form.elements.transcript_upload.value = await upload(transcript, "transcript", form, () => {});
        form.elements.transcript_kind.value = suffix;
      }
      status.textContent = "Validating and importing…";
      form.submit();
    } catch (error) {
      status.textContent = error.message;
      form.querySelector("button[type=submit]").disabled = false;
      form.elements.audio_upload.value = "";
    }
  });
}

function seekToAnchor() {
  // Search results link to #s000123: position the player there without autoplaying.
  const player = document.getElementById("player");
  const id = decodeURIComponent(window.location.hash.slice(1));
  if (!player || !/^s\d{6,}$/.test(id)) return;
  const link = document.querySelector(`tr[id="${id}"] a.seek[data-start]`);
  const start = link ? Number(link.dataset.start) : NaN;
  if (Number.isFinite(start)) {
    const apply = () => { player.currentTime = start; };
    if (player.readyState >= 1) apply();
    else player.addEventListener("loadedmetadata", apply, { once: true });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  seekLinks();
  seekToAnchor();
  importForm();
});
