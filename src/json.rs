//! A small JSON reader (manifest and test data). Strings are kept as UTF-16 code units, like
//! JavaScript strings, so escaped lone surrogates (`"\ud800"`) survive. Numbers are parsed
//! with Rust's correctly rounded `f64` parser, which gives the same values as `JSON.parse`.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(Vec<u16>),
    Arr(Vec<Value>),
    Obj(HashMap<String, Value>),
}

impl Value {
    pub fn get(&self, key: &str) -> &Value {
        match self {
            Value::Obj(m) => m.get(key).unwrap_or(&Value::Null),
            _ => &Value::Null,
        }
    }
    pub fn has(&self, key: &str) -> bool {
        matches!(self, Value::Obj(m) if m.contains_key(key))
    }
    pub fn num(&self) -> f64 {
        match self {
            Value::Num(n) => *n,
            _ => panic!("JSON: expected a number, got {:?}", self),
        }
    }
    pub fn usize(&self) -> usize {
        self.num() as usize
    }
    pub fn bool(&self) -> bool {
        matches!(self, Value::Bool(true))
    }
    pub fn arr(&self) -> &[Value] {
        match self {
            Value::Arr(a) => a,
            _ => panic!("JSON: expected an array, got {:?}", self),
        }
    }
    pub fn utf16(&self) -> &[u16] {
        match self {
            Value::Str(s) => s,
            _ => panic!("JSON: expected a string, got {:?}", self),
        }
    }
    /// The string, with lone surrogates replaced by U+FFFD.
    pub fn string(&self) -> String {
        String::from_utf16_lossy(self.utf16())
    }
}

pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("JSON: trailing data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("JSON: {} at byte {}", what, self.i))
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.s.get(self.i) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.word("true", Value::Bool(true)),
            Some(b'f') => self.word("false", Value::Bool(false)),
            Some(b'n') => self.word("null", Value::Null),
            Some(_) => self.number(),
            None => self.err("unexpected end"),
        }
    }

    fn word(&mut self, w: &str, v: Value) -> Result<Value, String> {
        if self.s[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            self.err("invalid literal")
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
            self.i += 1;
        }
        let t = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        t.parse::<f64>().map(Value::Num).or_else(|_| self.err("invalid number"))
    }

    fn hex4(&mut self) -> Result<u16, String> {
        if self.i + 4 > self.s.len() {
            return self.err("bad \\u escape");
        }
        let t = std::str::from_utf8(&self.s[self.i..self.i + 4]).map_err(|e| e.to_string())?;
        let v = u16::from_str_radix(t, 16).or_else(|_| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<Vec<u16>, String> {
        self.i += 1; // opening quote
        let mut out: Vec<u16> = Vec::new();
        loop {
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                self.i += 1;
            }
            let chunk = std::str::from_utf8(&self.s[start..self.i]).map_err(|e| e.to_string())?;
            out.extend(chunk.encode_utf16());
            match self.s.get(self.i) {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = match self.s.get(self.i) {
                        Some(c) => *c,
                        None => return self.err("unterminated string"),
                    };
                    self.i += 1;
                    match c {
                        b'"' => out.push(0x22),
                        b'\\' => out.push(0x5C),
                        b'/' => out.push(0x2F),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'n' => out.push(0x0A),
                        b'r' => out.push(0x0D),
                        b't' => out.push(0x09),
                        b'u' => {
                            let u = self.hex4()?;
                            out.push(u);
                        }
                        _ => return self.err("bad escape"),
                    }
                }
                _ => return self.err("unterminated string"),
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Value::Arr(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Arr(out));
                }
                _ => return self.err("expected , or ]"),
            }
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.i += 1;
        let mut out = HashMap::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Value::Obj(out));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return self.err("expected a key");
            }
            let k = String::from_utf16_lossy(&self.string()?);
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return self.err("expected :");
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            out.insert(k, v);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Obj(out));
                }
                _ => return self.err("expected , or }"),
            }
        }
    }
}

/// Minimal JSON string escaping for output (UTF-16 input, lone surrogates as \uXXXX).
pub fn quote_utf16(s: &[u16]) -> String {
    let mut out = String::from("\"");
    for r in char::decode_utf16(s.iter().cloned()) {
        match r {
            Ok('"') => out.push_str("\\\""),
            Ok('\\') => out.push_str("\\\\"),
            Ok('\n') => out.push_str("\\n"),
            Ok('\r') => out.push_str("\\r"),
            Ok('\t') => out.push_str("\\t"),
            Ok(c) if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            Ok(c) => out.push(c),
            Err(e) => out.push_str(&format!("\\u{:04x}", e.unpaired_surrogate())),
        }
    }
    out.push('"');
    out
}

/// Base64 decoding (test data).
pub fn base64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    }
    let b: Vec<u8> = s.bytes().filter(|&c| c != b'=' && !c.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(b.len() * 3 / 4);
    for chunk in b.chunks(4) {
        let mut n = 0u32;
        for (j, &c) in chunk.iter().enumerate() {
            n |= val(c) << (18 - 6 * j);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    out
}
