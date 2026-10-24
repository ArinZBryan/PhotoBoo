"use client";

import { useCallback, useEffect, useState } from "react";

const src = (name) => `/api/image/${encodeURIComponent(name)}`;

function VoteControls({ votes, onAdd, onRemove }) {
  return (
    <div className="votes">
      <button onClick={onRemove} disabled={votes === 0} aria-label="Remove vote">
        −
      </button>
      <span>
        ♥ {votes} {votes === 1 ? "vote" : "votes"}
      </span>
      <button onClick={onAdd} aria-label="Add vote">
        +
      </button>
    </div>
  );
}

export default function Gallery() {
  const [images, setImages] = useState([]);
  const [selected, setSelected] = useState(new Set());
  const [open, setOpen] = useState(null); // index of the image in the viewer

  const load = useCallback(async () => {
    const res = await fetch("/api/images", { cache: "no-store" });
    const list = await res.json();
    setImages(list);
    // drop selections for files that no longer exist
    setSelected((prev) => {
      const names = new Set(list.map((i) => i.name));
      return new Set([...prev].filter((n) => names.has(n)));
    });
    setOpen((i) => (i === null ? null : Math.min(i, list.length - 1)));
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  const toggle = (name) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });

  const vote = async (name, action) => {
    const res = await fetch("/api/vote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, action }),
    });
    if (!res.ok) return;
    const { votes } = await res.json();
    setImages((prev) => prev.map((i) => (i.name === name ? { ...i, votes } : i)));
  };

  const step = useCallback(
    (d) => setOpen((i) => (i === null ? i : (i + d + images.length) % images.length)),
    [images.length]
  );

  useEffect(() => {
    if (open === null) return;
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(null);
      else if (e.key === "ArrowLeft") step(-1);
      else if (e.key === "ArrowRight") step(1);
      else if (e.key === "v" || e.key === "+") vote(images[open].name, "add");
      else if (e.key === "-") vote(images[open].name, "remove");
      else if (e.key === " ") {
        e.preventDefault();
        toggle(images[open].name);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const current = open !== null ? images[open] : null;

  return (
    <main>
      <header>
        <h1>PhotoBoo Gallery</h1>
        <div className="actions">
          <span>
            {selected.size} of {images.length} selected
          </span>
          <button onClick={() => setSelected(new Set(images.map((i) => i.name)))}>
            Select all
          </button>
          <button onClick={() => setSelected(new Set())} disabled={!selected.size}>
            Clear
          </button>
        </div>
      </header>
      {images.length === 0 ? (
        <p className="empty">No captures yet.</p>
      ) : (
        <div className="grid">
          {images.map(({ name, votes }, idx) => (
            <div key={name} className="card">
              <div className={"tile" + (selected.has(name) ? " selected" : "")}>
                <button
                  className="pick"
                  onClick={() => toggle(name)}
                  onDoubleClick={() => setOpen(idx)}
                  aria-pressed={selected.has(name)}
                  title={`${name} (click to select, double-click to enlarge)`}
                >
                  <img src={src(name)} alt={name} loading="lazy" />
                </button>
                <span className="check">✓</span>
                <button className="expand" onClick={() => setOpen(idx)} aria-label="View larger">
                  ⤢
                </button>
              </div>
              <VoteControls
                votes={votes}
                onAdd={() => vote(name, "add")}
                onRemove={() => vote(name, "remove")}
              />
            </div>
          ))}
        </div>
      )}

      {current && (
        <div className="viewer" onClick={() => setOpen(null)}>
          <button className="close" onClick={() => setOpen(null)} aria-label="Close">
            ✕
          </button>
          <button
            className="nav prev"
            onClick={(e) => (e.stopPropagation(), step(-1))}
            aria-label="Previous"
          >
            ‹
          </button>
          <figure onClick={(e) => e.stopPropagation()}>
            <img src={src(current.name)} alt={current.name} />
            <figcaption>
              <span>
                {open + 1} / {images.length} · {current.name}
              </span>
              <span className="viewer-actions">
                <VoteControls
                  votes={current.votes}
                  onAdd={() => vote(current.name, "add")}
                  onRemove={() => vote(current.name, "remove")}
                />
                <button
                  className={"sel" + (selected.has(current.name) ? " on" : "")}
                  onClick={() => toggle(current.name)}
                >
                  {selected.has(current.name) ? "✓ Selected" : "Select"}
                </button>
              </span>
            </figcaption>
            <small>← → navigate · + / - vote · Space select · Esc close</small>
          </figure>
          <button
            className="nav next"
            onClick={(e) => (e.stopPropagation(), step(1))}
            aria-label="Next"
          >
            ›
          </button>
        </div>
      )}
    </main>
  );
}
