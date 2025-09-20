'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for LLM prompting
'''

MEMELANG_VER = 9.02

import random, re, json, operator
from typing import List, Iterator, Iterable, Dict, Tuple, Any, Union

Axis, Memelang = int, str

SIGIL, WILD, MSAME, VSAME, EOF =  '$', '_', '^', '@', None
SEP_LIMIT, SEP_VCTR, SEP_MTRX = ' ', ';', ';;'
SEP_VCTR_PRETTY, SEP_MTRX_PRETTY = ' ; ', ' ;;\n'
ELIDE_LIMIT = ['PRIOR', 'VSAME']
LEFT, RIGHT = 0, 1


TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS "John \"Jack\" Kennedy"
	('MDTBL',		r'-*\|'),
	('SEP_MTRX',	re.escape(SEP_MTRX)),
	('SEP_VCTR',	re.escape(SEP_VCTR)),
	('SEP_LIMIT',	r'\s+'),

	('POW',			r'\*\*'),
	('MUL',			r'\*'),
	('ADD',			r'\+'),
	('DIV',			r'\/'),
	('MOD',			r'\%'),
	('TSQ',			r'@@'),
	('L2',			r'<->'),
	('COS',			r'<=>'),
	('IP',			r'<#>'),
	
	('GE',			r'>='),
	('LE',			r'<='),
	('NOT',			r'!=?'),
	('EQL',			r'='),
	('GT',			r'>'),
	('LT',			r'<'),
	
	('META',		r'`'),
	
	('WILD',		re.escape(WILD)),		# WILDCARD, MATCHES WHOLE VALUE, NEVER QUOTE
	('MSAME',		re.escape(MSAME)),		# REFERENCES (MTRX_AXIS-1, VCTR_AXIS=-1, LIMIT_AXIS)
	('VSAME',		re.escape(VSAME)),		# REFERENCES (MTRX_AXIS,   VCTR_AXIS-1,  LIMIT_AXIS)
	('VAR',			r'\$[A-Za-z0-9]+'),
	('ALNUM',		r'[A-Za-z_][A-Za-z0-9_]*'), # ALPHANUMERICS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),

	('SUB',			r'\-'), # AFTER INT/FLOAT

	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

CMP_KINDS = {'EQL':{'STR','NUM','DATA'},'NOT':{'STR','NUM','DATA'},'GT':{'NUM'},'GE':{'NUM'},'LT':{'NUM'},'LE':{'NUM'}}
MOD_KINDS = {'META':{},'MUL':{'NUM'},'ADD':{'NUM'},'SUB':{'NUM'},'DIV':{'NUM'},'POW':{'NUM'},'L2':{'VEC'},'IP':{'VEC'},'COS':{'VEC'},'TSQ':{'TSQ'}}
DATUM_KINDS = {'ALNUM','QUOTE','INT','FLOAT','VAR','VSAME','MSAME','WILD'}
IGNORE_KINDS = {'COMMENT','MDTBL'}

EBNF = '''
TERM ::= [MOD] DATUM
EXPR ::= TERM {TERM}
COMP ::= [CMP] EXPR
LIMIT ::= COMP COMP {COMP}
VCTR ::= LIMIT {SEP_LIMIT LIMIT}
MTRX ::= VCTR {SEP_VCTR VCTR}
MEME ::= MTRX {SEP_MTRX MTRX}
'''

class Token():
	kind: list[str]
	lexeme: str
	datum: Union[str, float, int, list]
	def __init__(self, kind: str, lexeme: str):
		self.kind = [kind]
		self.lexeme = lexeme
		if kind == 'QUOTE': 	self.datum = json.loads(lexeme)
		elif kind == 'FLOAT': 	self.datum = float(lexeme)
		elif kind == 'INT':		self.datum = int(lexeme)
		else: 					self.datum = lexeme

	def dump(self) -> Union[str, float, int, list]: return self.datum
	def __str__(self) -> Memelang: return self.lexeme
	def __eq__(self, other): return isinstance(other, Token) and self.kind[0] == other.kind[0] and self.lexeme == other.lexeme


TOK_EQL = Token('EQL', '') # ELIDED '='
TOK_NOT = Token('NOT', '!')
TOK_GT = Token('GT', '>')
TOK_SEP_NONE = Token('SEP_NONE', '') # ELIDED
TOK_SEP_LIMIT = Token('SEP_LIMIT', SEP_LIMIT)
TOK_SEP_VCTR = Token('SEP_VCTR', SEP_VCTR)
TOK_SEP_MTRX = Token('SEP_MTRX', SEP_MTRX)

TOK_SEP_TERM = Token('SEP_TERM', '')
TOK_SEP_OR = Token('SEP_OR', '')
TOK_SEP_TRUE = Token('SEP_TRUE', '')


class Stream:
	def __init__(self, token: Iterable[Token]):
		self.token: Iterator[Token] = iter(token)
		self.buffer: List[Token] = []

	def peek(self, fwd: int = 1) -> Union[str, None]:
		while(len(self.buffer)<fwd):
			val = next(self.token, EOF)
			if val is EOF: return EOF
			self.buffer.append(val)
		return self.buffer[fwd-1].kind[0]
		
	def next(self) -> Token: 
		if not self.buffer:
			val = next(self.token, EOF)
			if val is EOF: raise SyntaxError('E_EOF')
			self.buffer.append(val)
		return self.buffer.pop(0)


class Node(list):
	opr: Token = TOK_EQL
	def __init__(self, *items: Union['Node', Token], opr:Token|None = None):
		super().__init__(items)
		if opr is not None: self.opr = opr

	def prepend(self, item):
		self.insert(0, item)

	def pad(self, padding:Union['Node', Token]) -> None:
		max_len = len(self[0])
		for idx, item in enumerate(self):
			diff = max_len - len(item)
			if diff>0: self[idx] += [padding] * diff
			elif diff<0: raise SyntaxError('E_PAD') # FIRST MUST BE LONGEST

	def dump(self) -> List: return [self.opr.dump(), [item.dump() for item in self]]
	def check(self) -> 'Node': 
		if len(self)==0: raise SyntaxError('E_NO_LIST')
		return self
	def __str__(self) -> Memelang: return self.opr.lexeme.join(map(str, self))

	@property
	def kind(self) -> list[str]:
		kind=[]
		for n in self: kind.extend(n.kind)
		return kind

# +5
class Term(Node):
	opr: Token = TOK_EQL
	def check(self) -> 'Term':
		if len(self)!=1: raise SyntaxError('E_NO_LIST')
		self.insert(0, Token('PRIOR', ''))
		return self


# +5+6/7
class Sqnc(Node):
	opr: Token = TOK_SEP_TERM


# >+5+6/7
class Pred(Node):
	opr: Token = TOK_EQL
	def check(self) -> 'Pred':
		if len(self)!=1: raise SyntaxError('E_NO_LIST')
		self.insert(0, Token('PRIOR', ''))
		return self

# (>+5+6/7 OR <+1+2+3)
class Junc(Node):
	opr: Token = TOK_SEP_OR


# Value must be (>+5+6/7 OR <+1+2+3)
class Limit(Node):
	opr: Token = TOK_SEP_TRUE
	def check(self) -> 'Pred':
	if len(self)!=2: raise SyntaxError('E_NO_LIST')


class Vector(Node):
	opr: Token = TOK_SEP_LIMIT
	def __str__(self) -> Memelang: return self.opr.lexeme.join(map(str, reversed(self)))


class Matrix(Node):
	opr: Token = TOK_SEP_VCTR


def lex(src: Memelang) -> Iterator[Token]:
	for m in MASTER_PATTERN.finditer(src):
		kind = m.lastgroup
		if kind in IGNORE_KINDS: continue
		if kind == 'MISMATCH': raise SyntaxError('E_TOK')
		yield Token(kind, m.group())


def parse(src: Memelang) -> Iterator[Matrix]:
	tokens = Stream(lex(src))
	bound_vars = []
	mtrx = Matrix()
	vctr = Vector()
	TERM_KINDS = set(MOD_KINDS)|DATUM_KINDS
	LIMIT_KINDS = set(CMP_KINDS)|TERM_KINDS

	while tokens.peek():

		# LIMIT: Single axis constraint
		preds = []
		pred = Pred()
		sqnc = Sqnc()

		while tokens.peek() in LIMIT_KINDS:

			if tokens.peek() in CMP_KINDS:
				pred.opr=tokens.next()

			while tokens.peek() in TERM_KINDS:
				term=Term()
				if tokens.peek() in MOD_KINDS: term.opr=tokens.next()
				if tokens.peek() in DATUM_KINDS: term.append(tokens.next())
				else: raise SyntaxError('E_EXPR_DATUM')
				sqnc.append(term.check())

			if sqnc: pred.append(sqnc.check())
			if pred: preds.append(pred.check())

			pred = Pred()
			sqnc = Sqnc()

		if preds:
			if len(preds)==1 or (preds[0].opr.lexeme!='' and preds[0][0][0].opr==TOK_EQL): preds.insert(0, Pred())
			for s in (LEFT, RIGHT):
				if not preds[s]:
					if ELIDE_LIMIT[s]: preds[s] = Pred(Sqnc(Term(Token(ELIDE_LIMIT[s], '')).check()).check()).check()
					else: raise SyntaxError(f'E_LIMIT_SIDE{s}')

			limit = Limit(preds[0], Junc(*preds[1:]))
				
			if len(mtrx)==0 and 'VSAME' in limit.kind: raise SyntaxError('E_VSAME_OOB')
			
			# FINALIZE LIMIT
			vctr.prepend(limit.check())
			continue

		# VCTR: Predunctive vector of axis constraints
		if tokens.peek() == 'SEP_VCTR':
			if vctr: mtrx.append(vctr.check())
			vctr = Vector()
			tokens.next()
			continue

		# MTRX: Predunctive matrix of axis constraints
		if tokens.peek() == 'SEP_MTRX':
			if vctr: mtrx.append(vctr.check())
			if mtrx: yield mtrx.check()
			vctr = Vector()
			mtrx = Matrix()
			tokens.next()
			continue

		if tokens.peek() == 'SEP_LIMIT':
			tokens.next()
			continue

		raise SyntaxError('E_TOK')

	if vctr: mtrx.append(vctr.check())
	if mtrx: yield mtrx.check()


class Meme(Node):
	opr: Token = TOK_SEP_MTRX
	results: List[List[List[Pred]]]
	bindings: Dict[str, Tuple[Axis, Axis, Axis]]
	src: Memelang

	def __init__(self, src: Memelang):
		self.src = src
		self.bindings = {}
		super().__init__(*parse(src))
		self.check()

	def check(self) -> 'Meme':
		for mtrx_idx, mtrx in enumerate(self):
			if not isinstance(mtrx, Matrix): raise TypeError('E_TYPE_MTRX')
			for vctr_idx, vctr in enumerate(mtrx):
				if not isinstance(vctr, Vector): raise TypeError('E_TYPE_VCTR')
				for limit_axis, limit in enumerate(vctr):
					if not isinstance(limit, Limit): raise TypeError('E_TYPE_LIMIT')
					# DO VAR BIND HERE
			self[mtrx_idx].pad(
				Limit(
						Pred(Sqnc(Term(Token('PRIOR','')).check()).check()).check(),
						Pred(Sqnc(Term(Token('VSAME','')).check()).check()).check(),
						opr=Token('EQL','=')
					).check()
			)

		self.results = [[[[] for limit in vctr] for vctr in mtrx] for mtrx in self]

		return self
