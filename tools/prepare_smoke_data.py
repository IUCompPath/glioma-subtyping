#!/usr/bin/env python3
"""Stage a handful of slides for the end-to-end smoke test.

Creates, under ``--out``:
    wsi/smoke/<slide>             symlinks to the chosen slides
    labels.csv                    slide_id,case_id,label  (DUMMY labels, cycling 0,1,2)
    splits/splits_0.csv           train = val = test = all slides (plumbing check only)

The labels and splits are placeholders so the pipeline can be exercised on a couple of
slides; metrics from the smoke test are meaningless.
"""
import argparse
import os
import sys

import pandas as pd

SLIDE_EXTS = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slides", nargs="+", required=True,
                   help="slide files, or a single directory (then use --n to pick the first N)")
    p.add_argument("--n", type=int, default=2, help="number of slides to take from a directory (default: 2)")
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args(argv)

    slides = args.slides
    if len(slides) == 1 and os.path.isdir(slides[0]):
        d = slides[0]
        slides = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(SLIDE_EXTS)][: args.n]
    if not slides:
        sys.exit("no slides found")

    wsi_dir = os.path.join(args.out, "wsi", "smoke")
    os.makedirs(wsi_dir, exist_ok=True)
    rows = []
    for i, path in enumerate(slides):
        name = os.path.basename(path)
        link = os.path.join(wsi_dir, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(path), link)
        slide_id = os.path.splitext(name)[0]
        rows.append({"slide_id": slide_id, "case_id": slide_id, "label": i % 3})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "labels.csv"), index=False)
    split_dir = os.path.join(args.out, "splits")
    os.makedirs(split_dir, exist_ok=True)
    pd.DataFrame({"train": df.slide_id, "val": df.slide_id, "test": df.slide_id}).to_csv(
        os.path.join(split_dir, "splits_0.csv"), index=False)
    print(f"staged {len(df)} slide(s) in {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
