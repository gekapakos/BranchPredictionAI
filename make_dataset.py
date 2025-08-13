import argparse, csv, pathlib, re

PAT_ADDR = re.compile(r"Branch Address:\s*([0-9a-fA-Fx]+),\s*Taken:\s*([01])")
PAT_HIST = re.compile(r"Branch History:\s*([01\s]*)")

def parse_trace(path, hist_len=20):
    """Yield tuples (addr:str, taken:int, history:str) where history is
       newest-bit-left, variable length ≤ hist_len."""
    with open(path, 'r') as f:
        lines = iter(f)
        for line in lines:
            m_addr = PAT_ADDR.match(line)
            if not m_addr:
                continue
            addr, taken = m_addr.group(1), int(m_addr.group(2))

            # Expect the next line to hold the history
            try:
                hist_line = next(lines)
            except StopIteration:
                raise ValueError("Missing Branch History after: " + line.strip())

            m_hist = PAT_HIST.match(hist_line)
            if not m_hist:
                raise ValueError("Malformed Branch History line: " + hist_line.strip())

            bits = m_hist.group(1).strip().split()
            bits = bits[:hist_len]           # truncate but **do not pad**
            yield addr.lower().lstrip('0x'), taken, ','.join(bits)

def main():
    ap = argparse.ArgumentParser(description="convert branch txt log to CSV (no padding)")
    ap.add_argument("infile",  type=pathlib.Path)
    ap.add_argument("outfile", type=pathlib.Path)
    ap.add_argument("--hist-len", type=int, default=20,
                    help="maximum history length kept (default 20)")
    args = ap.parse_args()

    with args.outfile.open('w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["branch_addr", "taken", "history"])
        for record in parse_trace(args.infile, args.hist_len):
            w.writerow(record)

    print(f"✓ Wrote {args.outfile}")

if __name__ == "__main__":
    main()