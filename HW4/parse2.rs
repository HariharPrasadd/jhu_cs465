//! probabilistic earley parser for homework 4

use std::collections::{HashMap, HashSet, VecDeque};
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

#[derive(Default)]
struct State {
    // transitions consume either a terminal or a completed nonterminal
    transitions: Vec<(String, usize)>,
    // accepting states charge the probability of the completed rule
    final_weight: Option<f64>,
    // split indexes avoid inspecting terminals absent from the sentence position
    terminals: HashMap<String, usize>,
    nonterminals: Vec<(String, usize)>,
}

struct Machine {
    lhs: String,
    states: Vec<State>,
    start: usize,
}

struct Grammar {
    start: String,
    // each nonterminal owns one minimized weighted automaton
    machines: Vec<Machine>,
    machine_ids: HashMap<String, usize>,
    // left parents connect a possible first symbol to parent machines
    left_parents: HashMap<String, Vec<usize>>,
}

impl Grammar {
    /// read rules and convert probabilities to negative log weights
    fn read(start: String, path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let mut groups: Vec<(String, Vec<(Vec<String>, f64)>)> = Vec::new();
        let mut group_ids = HashMap::new();

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

            let lhs = fields[1].to_owned();
            let group = *group_ids.entry(lhs.clone()).or_insert_with(|| {
                groups.push((lhs, Vec::new()));
                groups.len() - 1
            });
            groups[group].1.push((
                fields[2].split_whitespace().map(str::to_owned).collect(),
                -probability.log2(),
            ));
        }

        // compile every rule group into one minimized acyclic automaton
        let mut machines: Vec<Machine> = groups
            .into_iter()
            .map(|(lhs, rules)| Machine::build(lhs, rules))
            .collect();
        let machine_ids: HashMap<String, usize> = machines
            .iter()
            .enumerate()
            .map(|(id, machine)| (machine.lhs.clone(), id))
            .collect();

        // specialize transitions after every nonterminal is known
        for machine in &mut machines {
            for state in &mut machine.states {
                for (symbol, next) in &state.transitions {
                    if machine_ids.contains_key(symbol) {
                        state.nonterminals.push((symbol.clone(), *next));
                    } else {
                        state.terminals.insert(symbol.clone(), *next);
                    }
                }
            }
        }
        let mut left_parents: HashMap<String, Vec<usize>> = HashMap::new();
        for (machine_id, machine) in machines.iter().enumerate() {
            for (symbol, _) in &machine.states[machine.start].transitions {
                left_parents
                    .entry(symbol.clone())
                    .or_default()
                    .push(machine_id);
            }
        }
        Ok(Self {
            start,
            machines,
            machine_ids,
            left_parents,
        })
    }

    fn state(&self, key: Key) -> &State {
        &self.machines[key.machine].states[key.state]
    }
}

impl Machine {
    fn build(lhs: String, rules: Vec<(Vec<String>, f64)>) -> Self {
        // first build a trie so common rule prefixes share states
        let mut trie = vec![State::default()];
        for (rhs, weight) in rules {
            let mut state = 0;
            for symbol in rhs {
                let next = trie[state]
                    .transitions
                    .iter()
                    .find(|(label, _)| label == &symbol)
                    .map(|(_, next)| *next);
                state = match next {
                    Some(next) => next,
                    None => {
                        let next = trie.len();
                        trie.push(State::default());
                        trie[state].transitions.push((symbol, next));
                        next
                    }
                };
            }
            trie[state].final_weight = Some(
                trie[state]
                    .final_weight
                    .map_or(weight, |old| old.min(weight)),
            );
        }

        // merge states with identical weighted futures to minimize the trie
        let mut states = Vec::new();
        let mut signatures: HashMap<(Option<u64>, Vec<(String, usize)>), usize> = HashMap::new();
        let start = Self::minimize(0, &trie, &mut states, &mut signatures);
        Self { lhs, states, start }
    }

    fn minimize(
        old: usize,
        trie: &[State],
        states: &mut Vec<State>,
        signatures: &mut HashMap<(Option<u64>, Vec<(String, usize)>), usize>,
    ) -> usize {
        // children are minimized before their parent is interned
        let mut transitions: Vec<(String, usize)> = trie[old]
            .transitions
            .iter()
            .map(|(symbol, next)| {
                (
                    symbol.clone(),
                    Self::minimize(*next, trie, states, signatures),
                )
            })
            .collect();
        transitions.sort();
        let signature = (
            trie[old].final_weight.map(f64::to_bits),
            transitions.clone(),
        );
        if let Some(&state) = signatures.get(&signature) {
            return state;
        }
        let state = states.len();
        states.push(State {
            transitions,
            final_weight: trie[old].final_weight,
            terminals: HashMap::new(),
            nonterminals: Vec::new(),
        });
        signatures.insert(signature, state);
        state
    }
}

/// structural identity of an item within one chart column
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Key {
    // the machine state replaces a set of similar dotted rules
    machine: usize,
    state: usize,
    // the end position is supplied by the containing column
    start: usize,
}

impl Key {
    fn advance(self, state: usize) -> Self {
        Self { state, ..self }
    }

    fn show(&self, grammar: &Grammar) -> String {
        let machine = &grammar.machines[self.machine];
        let next: Vec<&str> = grammar
            .state(*self)
            .transitions
            .iter()
            .map(|(symbol, _)| symbol.as_str())
            .collect();
        format!(
            "({}, {} automaton state {} → {{{}}})",
            self.start,
            machine.lhs,
            self.state,
            next.join(", ")
        )
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
    // customers are grouped by the nonterminal they are waiting for
    customers: HashMap<String, Vec<usize>>,
    // completions merge internal analyses with the same category and span
    completions: HashMap<(usize, usize), f64>,
}

impl Column {
    /// insert an item or replace its derivation when the new weight is lower
    fn push(&mut self, key: Key, weight: f64, back: Option<BackPointer>, grammar: &Grammar) {
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
            for (symbol, _) in &grammar.state(key).nonterminals {
                self.customers
                    .entry(symbol.clone())
                    .or_default()
                    .push(index);
            }
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
    // each position maps a machine to viable first transitions
    left_corners: Vec<HashMap<usize, HashSet<String>>>,
    // counts retain the starter code profiling behavior
    counts: [usize; 3],
    progress: bool,
    verbose: bool,
}

impl<'a> Chart<'a> {
    fn build(tokens: Vec<String>, grammar: &'a Grammar, progress: bool, verbose: bool) -> Self {
        // a sentence with n tokens has n plus one chart columns
        let mut left_corners: Vec<_> = tokens
            .iter()
            .map(|token| Self::left_corners_for(grammar, token))
            .collect();
        left_corners.push(HashMap::new());
        let mut chart = Self {
            columns: (0..=tokens.len()).map(|_| Column::default()).collect(),
            tokens,
            grammar,
            left_corners,
            counts: [0; 3],
            progress,
            verbose,
        };
        chart.run();
        chart
    }

    fn left_corners_for(grammar: &Grammar, token: &str) -> HashMap<usize, HashSet<String>> {
        // walk upward from the word through every possible left parent
        let mut table: HashMap<usize, HashSet<String>> = HashMap::new();
        let mut seen = HashSet::new();
        let mut stack = vec![token.to_owned()];
        while let Some(child) = stack.pop() {
            if !seen.insert(child.clone()) {
                continue;
            }
            for &parent in grammar.left_parents.get(&child).into_iter().flatten() {
                table.entry(parent).or_default().insert(child.clone());
                stack.push(grammar.machines[parent].lhs.clone());
            }
        }
        table
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
                let state = self.grammar.state(item.key);
                let final_weight = state.final_weight;
                let at_start = item.key.state == self.grammar.machines[item.key.machine].start;
                let allowed = self.left_corners[end].get(&item.key.machine);
                let nonterminals: Vec<_> = state
                    .nonterminals
                    .iter()
                    .filter(|(symbol, _)| {
                        !at_start || allowed.map_or(false, |set| set.contains(symbol))
                    })
                    .cloned()
                    .collect();
                let terminal = self.tokens.get(end).and_then(|token| {
                    if !at_start || allowed.map_or(false, |set| set.contains(token)) {
                        state
                            .terminals
                            .get(token)
                            .map(|next| (token.clone(), *next))
                    } else {
                        None
                    }
                });

                // an accepting state may also have outgoing transitions
                if let Some(rule_weight) = final_weight {
                    let weight = item.weight + rule_weight;
                    let completion = (item.key.machine, item.key.start);
                    let improves = self.columns[end]
                        .completions
                        .get(&completion)
                        .map_or(true, |old| weight < *old);
                    if improves {
                        self.columns[end].completions.insert(completion, weight);
                        self.log(&item, "ATTACH");
                        self.attach(item_ref, &item, weight, end);
                    }
                }
                for (symbol, _) in nonterminals {
                    self.log(&item, "PREDICT");
                    self.predict(&symbol, end);
                }
                if let Some((terminal, next)) = terminal {
                    self.log(&item, "SCAN");
                    self.scan(item_ref, &item, end, &terminal, next);
                }
            }
        }
        if self.progress {
            eprintln!();
        }
    }

    fn predict(&mut self, nonterminal: &str, end: usize) {
        // prediction enters one shared machine instead of every individual rule
        if let Some(&machine) = self.grammar.machine_ids.get(nonterminal) {
            if !self.left_corners[end].contains_key(&machine) {
                return;
            }
            self.columns[end].push(
                Key {
                    machine,
                    state: self.grammar.machines[machine].start,
                    start: end,
                },
                0.0,
                None,
                self.grammar,
            );
            self.counts[0] += 1;
        }
    }

    fn scan(&mut self, item_ref: ItemRef, item: &Item, end: usize, terminal: &str, next: usize) {
        // scanning advances only when the next terminal matches the next token
        if end < self.tokens.len() && terminal == self.tokens[end] {
            self.columns[end + 1].push(
                item.key.advance(next),
                item.weight,
                Some(BackPointer::Scan(item_ref, self.tokens[end].clone())),
                self.grammar,
            );
            self.counts[1] += 1;
        }
    }

    fn attach(&mut self, child_ref: ItemRef, child: &Item, child_weight: f64, end: usize) {
        let lhs = self.grammar.machines[child.key.machine].lhs.as_str();
        let customer_ids = self.columns[child.key.start]
            .customers
            .get(lhs)
            .cloned()
            .unwrap_or_default();
        let customers: Vec<(ItemRef, Key, f64)> = customer_ids
            .into_iter()
            .map(|index| {
                let item = &self.columns[child.key.start].items[index];
                let next = self
                    .grammar
                    .state(item.key)
                    .nonterminals
                    .iter()
                    .find(|(symbol, _)| symbol == lhs)
                    .unwrap()
                    .1;
                (
                    (child.key.start, index),
                    item.key.advance(next),
                    item.weight,
                )
            })
            .collect();

        for (customer_ref, key, weight) in customers {
            self.columns[end].push(
                key,
                weight + child_weight,
                Some(BackPointer::Attach(customer_ref, child_ref)),
                self.grammar,
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
                    && self.grammar.state(item.key).final_weight.is_some()
                    && self.grammar.machines[item.key.machine].lhs == self.grammar.start
            })
            .min_by(|(_, left), (_, right)| {
                self.complete_weight(left)
                    .total_cmp(&self.complete_weight(right))
            })
            .map(|(index, _)| (end, index))
    }

    fn complete_weight(&self, item: &Item) -> f64 {
        item.weight + self.grammar.state(item.key).final_weight.unwrap_or(0.0)
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
        let lhs = &self.grammar.machines[item.key.machine].lhs;
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
            println!(
                "{}",
                chart.complete_weight(&chart.columns[root.0].items[root.1])
            );
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
