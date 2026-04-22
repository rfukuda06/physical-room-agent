// LogShell — shared chrome for the three scrollable log panels.
// Owns the card, the title bar (with total count), and the scroll viewport.
// Content is provided as children — each log renders its own rows.

"use client";

export default function LogShell({
  title,
  count,
  accent,
  children,
  className = "",
}: {
  title: string;
  count: number;
  accent?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`flex min-h-0 flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/60 ${className}`}
    >
      <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-4 py-2">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
            {title}
          </h3>
          <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[11px] text-zinc-400">
            {count}
          </span>
        </div>
        {accent && <div className="text-[11px] text-zinc-500">{accent}</div>}
      </div>
      {/* No scroll. Overflow is clipped by the parent's overflow-hidden.
          Newest entries are at the top of the list so visible content is
          always the most recent; older entries silently fall off the
          bottom of the viewport when they no longer fit. */}
      <div className="min-h-0 flex-1 overflow-hidden px-3 py-2">
        {count === 0 ? (
          <div className="py-6 text-center text-sm text-zinc-500">
            waiting…
          </div>
        ) : (
          children
        )}
      </div>
    </div>
  );
}
