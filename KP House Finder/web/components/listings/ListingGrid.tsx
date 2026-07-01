import type { Listing } from "@/lib/types";
import { ListingCard } from "./ListingCard";

export interface ListingGridProps {
  listings: Listing[];
  savedSet?: Set<string>;
  contactedSet?: Set<string>;
  onSave?: (id: string) => void;
  onContacted?: (id: string) => void;
  onRemove?: (id: string) => void;
  onRestore?: (id: string) => void;
  onEditPrice?: (id: string, price: number | null) => Promise<void> | void;
  onOpen?: (listing: Listing) => void;
}

export function ListingGrid({ listings, savedSet, contactedSet, onSave, onContacted, onRemove, onRestore, onEditPrice, onOpen }: ListingGridProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {listings.map((l) => (
        <ListingCard key={l.id} listing={l} saved={savedSet?.has(l.id) ?? false} contacted={contactedSet?.has(l.id) ?? false}
          onSave={onSave} onContacted={onContacted} onRemove={onRemove} onRestore={onRestore} onEditPrice={onEditPrice} onOpen={onOpen} />
      ))}
    </div>
  );
}
