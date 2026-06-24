import { describe, it, expect } from "vitest";
import { mergeLinks } from "./merge";
import type { Listing } from "./types";

const L = (id: string): Listing => ({ ...({} as Listing), id, link: null, url: null });

describe("mergeLinks", () => {
  it("attaches link/url from fb_posts by id", () => {
    const offers = [L("a"), L("b")];
    const posts = [{ id: "a", link: "L-a", url: "U-a" }, { id: "c", link: "L-c", url: "U-c" }];
    const out = mergeLinks(offers, posts);
    expect(out.find((r) => r.id === "a")).toMatchObject({ link: "L-a", url: "U-a" });
    expect(out.find((r) => r.id === "b")).toMatchObject({ link: null, url: null });
  });
});
