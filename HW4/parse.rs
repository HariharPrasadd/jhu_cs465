//! probabilistic earley parser for homework 4

use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

struct Args {
    // paths remain separate from parser options
    grammar: PathBuf,
    sentences: PathBuf,
    start: String,
    progress: bool,
    verbose: bool,
}

/// parse the command line used by the python starter
fn parse_args() -> Result<Args, String> {
    // flags may appear anywhere among the two positional paths
    let mut paths = Vec::new();
    let mut start = String::from("ROOT");
    let mut progress = false;
    let mut verbose = false;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-s" | "--start_symbol" => {
                start = args.next().ok_or("missing start symbol")?;
            }
            "--progress" => progress = true,
            "-v" | "--verbose" => verbose = true,
            "-q" | "--quiet" => verbose = false,
            "-h" | "--help" => {
                println!(
                    "Usage: parse [OPTIONS] GRAMMAR SENTENCES\n\
                     Options:\n\
                       -s, --start_symbol SYMBOL\n\
                           --progress\n\
                       -v, --verbose\n\
                       -q, --quiet"
                );
                process::exit(0);
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}")),
            _ => paths.push(PathBuf::from(arg)),
        }
    }

    if paths.len() != 2 {
        return Err(String::from("expected grammar and sentence files"));
    }
    Ok(Args {
        grammar: paths.remove(0),
        sentences: paths.remove(0),
        start,
        progress,
        verbose,
    })
}

struct Rule {
    lhs: String,
    rhs: Vec<String>,
    weight: f64,
}

struct Grammar {
    // all chart items share this single grammar allocation
    start: String,
    rules: Vec<Rule>,
    expansions: HashMap<String, Vec<usize>>,
}

impl Grammar {
    /// read rules and convert probabilities to negative log weights
    fn read(start: String, path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let mut grammar = Self {
            start,
            rules: Vec::new(),
            expansions: HashMap::new(),
        };

        for (line_number, source) in text.lines().enumerate() {
            // ignore blank lines and anything following a hash
            let line = source.split('#').next().unwrap_or("").trim_end();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != 3 {
                return Err(format!(
                    "{}:{}: expected three tab separated fields",
                    path.display(),
                    line_number + 1
                ));
            }
            let probability: f64 = fields[0].parse().map_err(|_| {
                format!(
                    "{}:{}: invalid probability",
                    path.display(),
                    line_number + 1
                )
            })?;
            if probability <= 0.0 {
                return Err(format!(
                    "{}:{}: probability must be positive",
                    path.display(),
                    line_number + 1
                ));
            }

            // store rules once and refer to them by integer index
            let id = grammar.rules.len();
            let lhs = fields[1].to_owned();
            grammar.rules.push(Rule {
                lhs: lhs.clone(),
                rhs: fields[2].split_whitespace().map(str::to_owned).collect(),
                weight: -probability.log2(),
            });
            grammar.expansions.entry(lhs).or_default().push(id);
        }
        Ok(grammar)
    }

    fn is_nonterminal(&self, symbol: &str) -> bool {
        self.expansions.contains_key(symbol)
    }
}

/// structural identity of an item within one chart column
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Key {
    // the rule and dot identify the dotted production
    rule: usize,
    dot: usize,
    // the end position is supplied by the containing column
    start: usize,
}

impl Key {
    fn next<'a>(&self, grammar: &'a Grammar) -> Option<&'a str> {
        grammar.rules[self.rule]
            .rhs
            .get(self.dot)
            .map(String::as_str)
    }

    fn advance(self) -> Self {
        Self {
            dot: self.dot + 1,
            ..self
        }
    }

    fn show(&self, grammar: &Grammar) -> String {
        let rule = &grammar.rules[self.rule];
        let mut rhs = rule.rhs.clone();
        rhs.insert(self.dot, String::from("·"));
        format!("({}, {} → {})", self.start, rule.lhs, rhs.join(" "))
    }
}

type ItemRef = (usize, usize);

/// one step used to reconstruct the best derivation
#[derive(Clone)]
enum BackPointer {
    Scan(ItemRef, String),
    Attach(ItemRef, ItemRef),
}

#[derive(Clone)]
struct Item {
    key: Key,
    // each record retains only its lightest known derivation
    weight: f64,
    back: Option<BackPointer>,
    // queued prevents duplicate work entries during improvements
    queued: bool,
}

/// a column keeps one record per key and a queue of records to process
#[derive(Default)]
struct Column {
    items: Vec<Item>,
    index: HashMap<Key, usize>,
    agenda: VecDeque<usize>,
}

impl Column {
    /// insert an item or replace its derivation when the new weight is lower
    fn push(&mut self, key: Key, weight: f64, back: Option<BackPointer>) {
        if let Some(&index) = self.index.get(&key) {
            // duplicate keys compete only on total derivation weight
            let item = &mut self.items[index];
            if weight < item.weight {
                item.weight = weight;
                item.back = back;
                // a processed improvement must be propagated again
                if !item.queued {
                    item.queued = true;
                    self.agenda.push_back(index);
                }
            }
        } else {
            // vector indices stay stable and serve as backpointer targets
            let index = self.items.len();
            self.items.push(Item {
                key,
                weight,
                back,
                queued: true,
            });
            self.index.insert(key, index);
            self.agenda.push_back(index);
        }
    }

    fn pop(&mut self) -> Option<(usize, Item)> {
        let index = self.agenda.pop_front()?;
        self.items[index].queued = false;
        Some((index, self.items[index].clone()))
    }
}

struct Chart<'a> {
    tokens: Vec<String>,
    grammar: &'a Grammar,
    // columns make the item end position implicit
    columns: Vec<Column>,
    // counts retain the starter code profiling behavior
    counts: [usize; 3],
    progress: bool,
    verbose: bool,
}

impl<'a> Chart<'a> {
    fn build(tokens: Vec<String>, grammar: &'a Grammar, progress: bool, verbose: bool) -> Self {
        // a sentence with n tokens has n plus one chart columns
        let mut chart = Self {
            columns: (0..=tokens.len()).map(|_| Column::default()).collect(),
            tokens,
            grammar,
            counts: [0; 3],
            progress,
            verbose,
        };
        chart.run();
        chart
    }

    fn run(&mut self) {
        // start by predicting every root production at position zero
        let start = self.grammar.start.clone();
        self.predict(&start, 0);

        // items added to the current column are processed by the same loop
        for end in 0..self.columns.len() {
            if self.progress {
                eprint!("\rProgress: {}/{}", end + 1, self.columns.len());
            }
            while let Some((index, item)) = self.columns[end].pop() {
                let item_ref = (end, index);
                match item.key.next(self.grammar).map(str::to_owned) {
                    None => {
                        self.log(&item, "ATTACH");
                        self.attach(item_ref, &item, end);
                    }
                    Some(symbol) if self.grammar.is_nonterminal(&symbol) => {
                        self.log(&item, "PREDICT");
                        self.predict(&symbol, end);
                    }
                    Some(_) => {
                        self.log(&item, "SCAN");
                        self.scan(item_ref, &item, end);
                    }
                }
            }
        }
        if self.progress {
            eprintln!();
        }
    }

    fn predict(&mut self, nonterminal: &str, end: usize) {
        // cloning these few indices avoids borrowing the grammar while mutating the chart
        let rules = self
            .grammar
            .expansions
            .get(nonterminal)
            .cloned()
            .unwrap_or_default();
        for rule in rules {
            self.columns[end].push(
                Key {
                    rule,
                    dot: 0,
                    start: end,
                },
                self.grammar.rules[rule].weight,
                None,
            );
            self.counts[0] += 1;
        }
    }

    fn scan(&mut self, item_ref: ItemRef, item: &Item, end: usize) {
        // scanning advances only when the next terminal matches the next token
        if end < self.tokens.len() && item.key.next(self.grammar) == Some(self.tokens[end].as_str())
        {
            self.columns[end + 1].push(
                item.key.advance(),
                item.weight,
                Some(BackPointer::Scan(item_ref, self.tokens[end].clone())),
            );
            self.counts[1] += 1;
        }
    }

    fn attach(&mut self, child_ref: ItemRef, child: &Item, end: usize) {
        let lhs = self.grammar.rules[child.key.rule].lhs.as_str();
        // question 3 deliberately uses a linear search for customers
        let customers: Vec<(ItemRef, Key, f64)> = self.columns[child.key.start]
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.key.next(self.grammar) == Some(lhs))
            .map(|(index, item)| ((child.key.start, index), item.key, item.weight))
            .collect();

        for (customer_ref, key, weight) in customers {
            self.columns[end].push(
                key.advance(),
                weight + child.weight,
                Some(BackPointer::Attach(customer_ref, child_ref)),
            );
            self.counts[2] += 1;
        }
    }

    /// choose the lightest complete start item spanning the sentence
    fn best(&self) -> Option<ItemRef> {
        let end = self.columns.len() - 1;
        self.columns[end]
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                item.key.start == 0
                    && item.key.next(self.grammar).is_none()
                    && self.grammar.rules[item.key.rule].lhs == self.grammar.start
            })
            .min_by(|(_, left), (_, right)| left.weight.total_cmp(&right.weight))
            .map(|(index, _)| (end, index))
    }

    /// follow backpointers to recover the children before an item's dot
    fn children(&self, item_ref: ItemRef) -> Vec<String> {
        let item = &self.columns[item_ref.0].items[item_ref.1];
        match &item.back {
            None => Vec::new(),
            Some(BackPointer::Scan(previous, terminal)) => {
                let mut children = self.children(*previous);
                children.push(terminal.clone());
                children
            }
            Some(BackPointer::Attach(previous, child)) => {
                let mut children = self.children(*previous);
                children.push(self.tree(*child));
                children
            }
        }
    }

    fn tree(&self, item_ref: ItemRef) -> String {
        // complete items render as the lhs followed by reconstructed children
        let item = &self.columns[item_ref.0].items[item_ref.1];
        let lhs = &self.grammar.rules[item.key.rule].lhs;
        format!("({lhs} {})", self.children(item_ref).join(" "))
    }

    fn log(&self, item: &Item, action: &str) {
        if self.verbose {
            eprintln!(
                "{} weight={} => {action}",
                item.key.show(self.grammar),
                item.weight
            );
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    // one grammar is reused for every nonblank sentence
    let grammar = Grammar::read(args.start, &args.grammar)?;
    let input = fs::read_to_string(&args.sentences)
        .map_err(|error| format!("could not read {}: {error}", args.sentences.display()))?;

    for line in input.lines() {
        let sentence = line.trim();
        if sentence.is_empty() {
            continue;
        }
        if args.verbose {
            eprintln!("{}\nParsing sentence: {sentence}", "=".repeat(70));
        }

        // input sentences are already tokenized with whitespace
        let tokens = sentence.split_whitespace().map(str::to_owned).collect();
        let chart = Chart::build(tokens, &grammar, args.progress, args.verbose);
        if let Some(root) = chart.best() {
            println!("{}", chart.tree(root));
            println!("{}", chart.columns[root.0].items[root.1].weight);
        } else {
            println!("NONE");
        }
        if args.verbose {
            eprintln!(
                "Profile of work done: PREDICT={}, SCAN={}, ATTACH={}",
                chart.counts[0], chart.counts[1], chart.counts[2]
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("error: {error}");
        process::exit(2);
    }
}
