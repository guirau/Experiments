import type { Listing } from "@/lib/types";
const HIDE = new Set(["id", "source_table", "parser_version", "raw_text"]);
export function ListingDetails({ listing }: { listing: Listing }) {
  const entries = Object.entries(listing).filter(([k, v]) => !HIDE.has(k) && v !== null && v !== "");
  return (
    <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 border-t pt-3 text-xs" style={{ borderColor: "var(--line)" }}>
      {entries.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-2">
          <dt style={{ color: "var(--muted)" }}>{k}</dt>
          <dd className="text-right font-medium">{String(v)}</dd>
        </div>
      ))}
      {listing.raw_text && <div className="col-span-2 mt-2"><dt style={{ color: "var(--muted)" }}>raw_text</dt><dd className="whitespace-pre-wrap">{listing.raw_text}</dd></div>}
    </dl>
  );
}
