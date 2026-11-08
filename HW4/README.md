# Homework 4 structure

The baseline parser is implemented in `parse.rs` and launched through
`parse.py`. The optimized parser is implemented in `parse2.rs` and launched
through `parse2.py`. On macOS, the wrappers compile their Rust source with
optimization and place the temporary executable outside this folder.
Gradescope does not expose `rustc`, so the wrappers use the bundled static
x86-64 Linux executables `parse-linux-x86_64` and `parse2-linux-x86_64` when
running on Linux.

`wallstreet.par` contains the optimized parser's required output for every
sentence in `wallstreet.sen`. 

Run either parser with a grammar followed by a sentence file:

```sh
./parse.py foo.gr foo.sen
./parse2.py foo.gr foo.sen
```

The programming submission consists of `parse.py`, `parse.rs`,
`parse-linux-x86_64`, `parse2.py`, `parse2.rs`, `parse2-linux-x86_64`, and
`wallstreet.par`.
