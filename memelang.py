'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs

NEVER SPACES BETWEEN COMPARATOR AND VALUES

1. MEMELANG USES AXES, HIGH -> LOW
3=Table 2=Primary_Key 1=Column 0=Value

2. EXAMPLE QUERY
MEMELANG: roles _ actor "Mark Hamill",Mark ; movie _ ; rating >4 ;;
SQL COLS: SELECT actor, movie, rating FROM roles WHERE actor IN ('Mark Hamill', 'Mark') AND rating > 4
SQL MEME: SELECT CONCAT_WS(' ', 'roles', t0.id, 'actor', t0.actor, ';', 'movie', t0.movie, ';', 'rating', t0.rating, ';;') AS meme FROM roles AS t0 WHERE t0.actor IN ('Mark Hamill', 'Mark') AND t0.rating > 4

3. EXAMPLE JOIN
MEMELANG: roles _ actor "Mark Hamill" ; movie _ ; !@ @ @ ; actor _ ;;
SQL COLS: SELECT t0.actor, t0.movie, t1.movie, t1.actor FROM roles AS t0, roles AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.id != t0.id AND t1.movie = t0.movie
SQL MEME: SELECT CONCAT_WS(' ', 'roles', t0.id, 'actor', t0.actor, ';', 'movie', t0.movie, ';', t1.id, 'movie', t1.movie, ';', 'actor', t1.actor, ';;' ) AS meme FROM roles AS t0, roles AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.id != t0.id AND t1.movie = t0.movie

4. EXAMPLE TABLE JOIN WHERE ACTOR NAME = MOVIE TITLE
MEMELANG: actors _ age >21; name _ ; roles _ title @ ;;
SQL COLS: SELECT t0.name, t0.age, t1.title FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t1.title = t0.name;
SQL MEME: SELECT CONCAT_WS(' ', 'actors', t0.id, 'age', t0.age, ';', 'name', t0.name, ';', 'roles', t1.id, 'title', t1.title, ';;' ) AS meme FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t1.title = t0.name
'''

MEMELANG_VER = 9.04

import random, re, json, operator
from typing import List, Iterator, Iterable, Dict, Tuple, Any, Union

Axis, Memelang = int, str

SIGIL, VAL, MSAME, VSAME, EOF =  '$', '_', '^', '@', None
SEP_LIMIT, SEP_VCTR, SEP_MTRX, SEP_OR = ' ', ';', ';;', ','
SEP_VCTR_PRETTY, SEP_MTRX_PRETTY = ' ; ', ' ;;\n'
LEFT, RIGHT = 0, 1


TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOT',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS "John \"Jack\" Kennedy"
	('MTBL',		r'-*\|'),
	('VEC',			r'\[(?:-?\d+(?:\.\d+)?)(?:,-?\d+(?:\.\d+)?)*\]'),

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
	
	('VAL',			re.escape(VAL)),		# VALCARD, MATCHES WHOLE VALUE, NEVER QUOTE
	('MSAME',		re.escape(MSAME)),		# REFERENCES (MTRX-1, VCTR=-1, LIMIT)
	('VSAME',		re.escape(VSAME)),		# REFERENCES (MTRX,   VCTR-1,  LIMIT)
	('VAR',			r'\$[A-Za-z0-9_]+'),
	('ALNUM',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERICS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),

	('SUB',			r'\-'), # AFTER INT/FLOAT

	('SEP_MTRX',	re.escape(SEP_MTRX)),
	('SEP_VCTR',	re.escape(SEP_VCTR)),
	('SEP_OR',		re.escape(SEP_OR)),
	('SEP_LIMIT',	r'\s+'),

	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

CMP_KINDS = {'EQL':{'STR','NUM','DATA'},'NOT':{'STR','NUM','DATA'},'GT':{'NUM'},'GE':{'NUM'},'LT':{'NUM'},'LE':{'NUM'}}
MOD_KINDS = {'MUL':{'NUM'},'ADD':{'NUM'},'SUB':{'NUM'},'DIV':{'NUM'},'MOD':{'NUM'},'POW':{'NUM'},'L2':{'VEC'},'IP':{'VEC'},'COS':{'VEC'},'TSQ':{'TSQ'}}
DATUM_KINDS = {'ALNUM','QUOT','INT','FLOAT','VAR','VSAME','MSAME','VAL','VEC'}
IGNORE_KINDS = {'COMMENT','MTBL'}

EBNF = '''
TERM ::= DATUM [MOD DATUM]
SQNC ::= TERM {TERM}
JUNC ::= {SQNC} {SEP_OR {SQNC}}
LIMIT ::= [SQNC] [CMP] [JUNC]
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
		if kind == 'QUOT': 		self.datum = json.loads(lexeme)
		elif kind == 'VEC': 	self.datum = json.loads(lexeme)
		elif kind == 'FLOAT': 	self.datum = float(lexeme)
		elif kind == 'INT':		self.datum = int(lexeme)
		else: 					self.datum = lexeme

	def dump(self) -> Union[str, float, int, list]: return self.datum
	def __str__(self) -> Memelang: return self.lexeme
	def __eq__(self, other): return isinstance(other, Token) and self.kind[0] == other.kind[0] and self.lexeme == other.lexeme


TOK_EQL = Token('EQL', '') # ELIDED '='
TOK_NOT = Token('NOT', '!')
TOK_GT = Token('GT', '>')
TOK_SEP_LIMIT = Token('SEP_LIMIT', SEP_LIMIT)
TOK_SEP_VCTR = Token('SEP_VCTR', SEP_VCTR)
TOK_SEP_MTRX = Token('SEP_MTRX', SEP_MTRX)
TOK_SEP_OR = Token('SEP_OR', SEP_OR)

TOK_SEP_TOK = Token('SEP_TOK', '')
TOK_SEP_TERM = Token('SEP_TERM', '')
TOK_SEP_PASS = Token('SEP_PASS', '')


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
	opr: Token = TOK_SEP_TOK


# 5+6/7
class Sqnc(Node):
	opr: Token = TOK_SEP_TERM


# (5+6/7 OR 1+2+3)
class Junc(Node):
	opr: Token = TOK_SEP_OR


# Value > (5+6/7 OR 1+2+3)
class Limit(Node):
	opr: Token = TOK_SEP_PASS
	def check(self) -> 'Limit':
		if len(self)<1 or len(self)>2: raise SyntaxError('E_NO_LIST')
		return self


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
	mtrx, vctr = Matrix(), Vector()
	LIMIT_KINDS = set(CMP_KINDS)|{'VAL'}|DATUM_KINDS

	while tokens.peek():

		# LIMIT: Single axis constraint
		limit = Limit(Sqnc(), Junc())
		side = None

		while tokens.peek() in LIMIT_KINDS:

			# CMP
			if tokens.peek() in CMP_KINDS:
				if side == RIGHT: raise SyntaxError('E_CMP_EXTRA')
				side, limit.opr = RIGHT, tokens.next()
				if tokens.peek() == 'SEP_LIMIT': raise SyntaxError('E_CMP_SPACE'); # NEVER SPACES INSIDE AXIS BETWEEN COMPARATOR AND VALUES

			# Start LEFT or RIGHT
			elif side is None:
				if tokens.peek() == 'VAL': side = LEFT
				else: side, limit.opr = RIGHT, TOK_EQL

			# SQNC
			sqnc = Sqnc()
			while tokens.peek() in DATUM_KINDS:
				if tokens.peek() not in DATUM_KINDS: raise SyntaxError('E_EXPR_DATUM')
				term=Term(tokens.next())
				if tokens.peek() in MOD_KINDS:
					term.opr=tokens.next()
					if tokens.peek() not in DATUM_KINDS: raise SyntaxError('E_EXPR_DATUM')
					term.append(tokens.next())

				sqnc.append(term.check())

			if sqnc: limit[side].append(sqnc.check())

			if tokens.peek() == 'SEP_OR': tokens.next()

		if side is not None:
			if len(mtrx)==0 and 'VSAME' in limit.kind: raise SyntaxError('E_VSAME_OOB')			
			vctr.prepend(limit.check())
			continue

		# VCTR
		if tokens.peek() == 'SEP_VCTR':
			if vctr: mtrx.append(vctr.check())
			vctr = Vector()
			tokens.next()
			continue

		# MTRX
		if tokens.peek() == 'SEP_MTRX':
			if vctr: mtrx.append(vctr.check())
			if mtrx: yield mtrx.check()
			mtrx, vctr = Matrix(), Vector()
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
	results: List[List[List[Junc]]]
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
						Sqnc(Term(Token('VAL','')).check()).check(),
						Junc(Sqnc(Term(Token('VSAME','')).check()).check()).check(),
						opr=Token('EQL','=')
					).check()
			)

		self.results = [[[[] for limit in vctr] for vctr in mtrx] for mtrx in self]

		return self


# GENERATE RANDOM MEMELANG DATA
class Fuzz():
	@staticmethod
	def datum(kind:str) -> Memelang:
		if kind=='ALNUM': return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(5))
		if kind=='QUOT': return json.dumps(''.join(random.choice(' -_+,./<>[]{}\'"!@#$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(10)))
		if kind=='INT': return str(random.randint(-9, 9))
		if kind=='FLOAT': return str(random.uniform(-9, 9))
		if kind=='VAR': return SIGIL + Fuzz.datum('ALNUM')

	@staticmethod
	def limit(bindings: List[str]|None = None) -> Memelang:
		if not bindings: bindings = []

		data: Memelang = ''

		comp = random.choice(['=','!=','>','<','<=','>='])

		# LEFT
		explicit_left = random.randint(0, 1)
		if explicit_left: 
			data += VAL
			second_term = random.randint(0, 1)
			if comp in {'>','<','>=','<='} and second_term:
				data += random.choice(['+','-','/','*','**']) + Fuzz.datum('INT')

		data+=comp

		# RIGHT
		if comp in {'=','!=','!'}:
			data_list_len = random.randint(1, 5)
			data_list: List[Any] = []
			for _ in range(data_list_len):
				datum_type = random.randint(1, 10)
				if datum_type == 1:  data_list.append(Fuzz.datum('QUOT'))
				elif datum_type == 2:  data_list.append(Fuzz.datum('INT'))
				elif datum_type == 3:  data_list.append(Fuzz.datum('FLOAT'))
				elif datum_type == 4 and bindings: data_list.append(random.choice(bindings))
				elif datum_type == 5 and VSAME in bindings: data_list.append(VSAME)
				else: data_list.append(Fuzz.datum('ALNUM'))
			data += SEP_OR.join(data_list)
		else:
			data += Fuzz.datum('FLOAT')
			second_term = random.randint(0, 1)
			if second_term: data += random.choice(['+','-','*','/']) + Fuzz.datum('FLOAT')

		return data

	@staticmethod
	def vector(limit_len:int = 4) -> Memelang:
		bindings, vector = [], []
		for i in range(limit_len):
			if i>0: bindings.append(VSAME)
			vector.append(Fuzz.limit(bindings))
		return SEP_LIMIT.join(vector) + SEP_VCTR_PRETTY

	@staticmethod
	def mtrx_table(col_len:int = 5) -> Memelang:
		return Fuzz.datum('ALNUM') + SEP_LIMIT + VAL + SEP_LIMIT + SEP_VCTR_PRETTY.join(Fuzz.datum('ALNUM') + Fuzz.limit() for _ in range(col_len)) + SEP_MTRX_PRETTY
