'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for LLM prompting
'''

MEMELANG_VER = 9.01

import random, re, json, operator
from typing import List, Iterator, Iterable, Dict, Tuple, Any, Union

Axis, Memelang = int, str

SIGIL, WILD, MSAME, VSAME, EOF =  '$', '_', '^', '@', None
SEP_COMP, SEP_DATA, SEP_VCTR, SEP_MTRX = ' ', ',', ';', ';;'
SEP_VCTR_PRETTY, SEP_MTRX_PRETTY = ' ; ', ' ;;\n'
ELIDE_COMP = ['VALUE', 'VSAME']
LEFT, RIGHT = 0, 1


TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS "John \"Jack\" Kennedy"
	('MDTBL',		r'-*\|'),
	('SEP_MTRX',	re.escape(SEP_MTRX)),
	('SEP_VCTR',	re.escape(SEP_VCTR)),
	('SEP_COMP',	r'\s+'),
	('SEP_DATA',	re.escape(SEP_DATA)),

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
	('MSAME',		re.escape(MSAME)),		# REFERENCES (MTRX_AXIS-1, VCTR_AXIS=-1, COMP_AXIS)
	('VSAME',		re.escape(VSAME)),		# REFERENCES (MTRX_AXIS,   VCTR_AXIS-1,  COMP_AXIS)
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
PAIR ::= [MOD] DATUM
EXPR ::= PAIR {PAIR}
DATA ::= EXPR {SEP_DATA EXPR}
COMP ::= [DATA] [CMP] [DATA]
VCTR ::= COMP {SEP_COMP COMP}
MTRX ::= VCTR {SEP_VCTR VCTR}
MEME ::= MTRX {SEP_MTRX MTRX}
'''

class Token():
	kind: str
	lexeme: str
	datum: Union[str, float, int, list]
	def __init__(self, kind: str, lexeme: str):
		self.kind = [kind]
		self.lexeme = lexeme
		if kind == 'QUOTE': 	self.datum = json.loads(lexeme)
		elif kind == 'FLOAT': 	self.datum = float(lexeme)
		elif kind == 'INT':		self.datum = int(lexeme)
		else: 					self.datum = lexeme

	@property
	def dump(self) -> str|float|int: return self.datum
	def __str__(self) -> Memelang: return self.lexeme
	def __eq__(self, other): return isinstance(other, Token) and self.kind[0] == other.kind[0] and self.lexeme == other.lexeme


TOK_EQL = Token('EQL', '') # ELIDED '='
TOK_NOT = Token('NOT', '!')
TOK_GT = Token('GT', '>')
TOK_SEP_NONE = Token('SEP_NONE', '') # ELIDED
TOK_SEP_DATA = Token('SEP_DATA', SEP_DATA)
TOK_SEP_COMP = Token('SEP_COMP', SEP_COMP)
TOK_SEP_VCTR = Token('SEP_VCTR', SEP_VCTR)
TOK_SEP_MTRX = Token('SEP_MTRX', SEP_MTRX)


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


class Pair(Node):
	opr: Token = TOK_EQL
	def check(self) -> 'Pair':
		if len(self)!=1: raise SyntaxError('E_NO_LIST')
		self.insert(0, Token('VALUE', ''))
		return self


class Expr(Node):
	opr: Token = TOK_SEP_NONE


class Data(Node):
	opr: Token = TOK_SEP_NONE


class Comp(Node):
	opr: Token = TOK_EQL
	def check(self) -> 'Comp':
		if len(self)!=2: raise SyntaxError('E_NO_LIST')
		if len(self[0])>1: raise SyntaxError('E_LEFT_DATA')
		return self


class Vector(Node):
	opr: Token = TOK_SEP_COMP
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
	MODDAT_KINDS = set(MOD_KINDS)|DATUM_KINDS
	COMP_KINDS = set(CMP_KINDS)|MODDAT_KINDS

	while tokens.peek():
		comp = Comp(Data(), Data())

		# COMP: Single axis constraint
		side = LEFT

		while tokens.peek() in COMP_KINDS:

			expr=Expr()
			while tokens.peek() in MODDAT_KINDS:
				pair=Pair()
				if tokens.peek() in MOD_KINDS: pair.opr=tokens.next()
				if tokens.peek() in DATUM_KINDS: pair.append(tokens.next())
				else: raise SyntaxError('E_EXPR_DATUM')
				expr.append(pair.check())

			if expr:
				comp[side].append(expr)

				while tokens.peek()=='SEP_DATA':
					if side==LEFT: raise SyntaxError('E_LEFT_DATA')
					comp[side].opr = tokens.next()
					if tokens.peek()=='SEP_COMP': raise SyntaxError('E_DATA_KIND_SPACE_AFTER_COMMA')
					if tokens.peek() not in MODDAT_KINDS: raise SyntaxError('E_DATA_KIND')
					
					expr=Expr()
					while tokens.peek() in MODDAT_KINDS:
						pair=Pair()
						if tokens.peek() in MOD_KINDS: pair.opr=tokens.next()
						if tokens.peek() in DATUM_KINDS: pair.append(tokens.next())
						else: raise SyntaxError('E_EXPR_DATUM')
						expr.append(pair.check())

					if expr: comp[side].append(expr)
			
			if tokens.peek() in CMP_KINDS:
				comp.opr=tokens.next()
				if side==LEFT: side=RIGHT
				else: raise SyntaxError('E_CMP')

		if comp[LEFT] and side==LEFT: comp[RIGHT], comp[LEFT] = comp[LEFT], Data()

		if any(d for d in comp):
			for s in (LEFT, RIGHT):
				if not comp[s]:
					if ELIDE_COMP[s]: comp[s] = Data(Expr(Pair(Token('VALUE', ''), Token(ELIDE_COMP[s], ''))))
					else: raise SyntaxError(f'E_COMP_SIDE{s}')
				
			if len(mtrx)==0 and 'VSAME' in comp.kind: raise SyntaxError('E_VSAME_OOB')
			
			# FINALIZE COMP
			vctr.prepend(comp.check())
			continue

		# VCTR: Conjunctive vector of axis constraints
		if tokens.peek() == 'SEP_VCTR':
			if vctr: mtrx.append(vctr.check())
			vctr = Vector()
			tokens.next()
			continue

		# MTRX: Conjunctive matrix of axis constraints
		if tokens.peek() == 'SEP_MTRX':
			if vctr: mtrx.append(vctr.check())
			if mtrx: yield mtrx.check()
			vctr = Vector()
			mtrx = Matrix()
			tokens.next()
			continue

		if tokens.peek() == 'SEP_COMP':
			tokens.next()
			continue

		raise SyntaxError('E_TOK')

	if vctr: mtrx.append(vctr.check())
	if mtrx: yield mtrx.check()


class Meme(Node):
	opr: Token = TOK_SEP_MTRX
	results: List[List[List[Data]]]
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
				for comp_axis, comp in enumerate(vctr):
					if not isinstance(comp, Comp): raise TypeError('E_TYPE_COMP')
					# DO VAR BIND HERE
			self[mtrx_idx].pad(Comp(opr=Token('EQL','='), Data(Expr(Pair(Token('VALUE', ''), Token('VALUE', '')))), Data(Expr(Pair(Token('VALUE', ''), Token('VSAME', ''))))))

		self.results = [[[[] for comp in vctr] for vctr in mtrx] for mtrx in self]

		return self
