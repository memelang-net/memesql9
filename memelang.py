'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs
MEMELANG USES AXES ORDERED HIGH TO LOW
ONE MORE OR WHITESPACES ***ALWAYS*** MEANS "NEW AXIS"
EXAMPLE: AXIS3 AXIS2 AXIS1 AXIS0
NEVER SPACE AROUND OPERATOR
NEVER SPACE BETWEEN COMPARATOR/COMMA/FUNC AND VALUES
EXAMPLE: roles _ actor "Mark Hamill"; rating >4; <5; movie _; !@ @ @; actor _;;
'''

MEMELANG_VER = 9.27

import random, re, json, sys
from typing import List, Iterator, Iterable, Dict, Tuple, Union

Memelang = str
Err = SyntaxError

ELIDE, SIGIL, VAL, MSAME, SAME, MODE, EOF =  '', '$', '_', '^', '@',  '%', None
SA, SV, SM, SF, OR, PRETTY = ' ', ';', ';;', ':', ',', ' '
L, R = 0, 1

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOT',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EMOTIC CHARS "John \"Jack\" Kennedy"
	('MTBL',		r'-*\|'),
	('EMB',			r'\[(?:-?\d+(?:\.\d+)?)(?:\s*,\s*-?\d+(?:\.\d+)?)*\]'), # JSON ARRAY OF DECS [0.1,0.2]
	('POW',			r'\*\*'),
	('MUL',			r'\*'),
	('ADD',			r'\+'),
	('DIV',			r'\/'),
	('MODE',		re.escape(MODE) + r'[a-z]+'),
	('MOD',			r'\%'),
	('L2',			r'<->'),
	('COS',			r'<=>'),
	('IP',			r'<#>'),
	('GE',			r'>='),
	('LE',			r'<='),
	('NOT',			r'!=?'),
	('EQL',			r'='),
	('GT',			r'>'),
	('LT',			r'<'),
	('SMLR',		r'~'),
	('VAL',			re.escape(VAL)),		# NEVER QUOTE
	('MSAME',		re.escape(MSAME)),		# REFERENCES (MAT-1, VEC=-1, LIMIT)
	('SAME',		re.escape(SAME)),		# REFERENCES (MAT,   VEC-1,  LIMIT)
	('VAR',			re.escape(SIGIL) + r'[A-Za-z0-9_]+'),
	('YMDHMS',		r'\d\d\d\d\-\d\d-\d\d\-\d\d:\d\d:\d\d'),	 	# YYYY-MM-DD-HH:MM:SS
	('YMD',			r'\d\d\d\d\-\d\d\-\d\d'),	 					# YYYY-MM-DD
	('ALNUM',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERICS ARE UNQUOTED
	('DEC',			r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('SUB',			r'\-'), # AFTER INT/DEC
	('SF',			re.escape(SF)),
	('SM',			re.escape(SM)),
	('SV',			re.escape(SV)),
	('OR',			re.escape(OR)),
	('SA',			r'\s+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

IGNORE_KINDS = {'COMMENT','MTBL'}
DELIDE = {'SAME':SAME,'MSAME':MSAME,'VAL': VAL,'EQL': '='}

D, Q, M = MODE+'d', MODE+'q', MODE+'m'
VOCAB = {
	D: { # DDL
		'CMP': {'EQL','NOT','GT','GE','LT','LE'},
		'MOD': {},
		'DAT': {'ALNUM','QUOT','INT','DEC','SAME','MSAME','VAL'},
		'FUNC': {'TYP','ROL','DESC'}
	},
	Q: { # DQL
		'CMP': {'EQL','NOT','GT','GE','LT','LE','SMLR'},
		'MOD': {'MUL','ADD','SUB','DIV','MOD','POW','L2','IP','COS'},
		'DAT': {'ALNUM','QUOT','INT','DEC','VAR','SAME','MSAME','VAL','EMB','YMD','YMDHMS'},
		'FUNC': {"grp","asc","dsc","sum","avg","min","max","cnt"}
	},
	M: { # META
		'CMP': {'EQL'},
		'MOD': {},
		'DAT': {'ALNUM','INT'}
	}
}

EBNF = '''
TERM ::= DAT [MOD DAT]
LEFT ::= [MOD DAT] {SF VAR|ALNUM}
RIGHT ::= TERM {OR TERM}
AXIS ::= LEFT [CMP RIGHT]
VEC ::= [MODE] AXIS {SA AXIS}
MAT ::= VEC {SV VEC}
MEME ::= MAT {SM MAT}
'''

class Token():
	kind: str
	kinds: List[str]
	lex: str
	delide: str
	dat: Union[str, float, int, list]
	def __init__(self, kind: str, lex: str):
		self.kind = kind
		self.kinds = [kind]
		self.lex = lex
		self.delide = DELIDE[kind] if lex == '' and kind in DELIDE else lex
		if kind=='QUOT': 	self.dat = json.loads(lex)
		elif kind=='EMB': 	self.dat = json.loads(lex)
		elif kind=='DEC': 	self.dat = float(lex)
		elif kind=='INT':	self.dat = int(lex)
		elif kind=='NULL':	self.dat = None
		else: 				self.dat = lex

	def dump(self) -> Tuple[str, Union[str, float, int, list, None]]: return (self.kind, self.dat)
	def __str__(self) -> Memelang: return self.lex
	def __eq__(self, other): return isinstance(other, Token) and self.kind==other.kind and self.lex==other.lex


TOK_NULL = Token('NULL', '')
TOK_EQL = Token('EQL', ELIDE)
TOK_NOT = Token('NOT', '!')
TOK_SA = Token('SA', SA)
TOK_SV = Token('SV', SV+PRETTY)
TOK_SM = Token('SM', SM+PRETTY)
TOK_OR = Token('OR', OR)
TOK_SF = Token('SF', SF)
TOK_SEP_TOK = Token('SEP_TOK', '')
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
		return self.buffer[fwd-1].kind
		
	def next(self) -> Token: 
		if not self.buffer:
			val = next(self.token, EOF)
			if val is EOF: raise Err('E_EOF')
			self.buffer.append(val)
		return self.buffer.pop(0)


class Node(list):
	opr: Token = TOK_NULL

	def __init__(self, *items): super().__init__(items)
	def prepend(self, item): self.insert(0, item)
	def dump(self) -> List: return [self.opr.dump(), [i.dump() for i in self]]
	def check(self) -> 'Node': 
		if len(self)==0: raise Err('E_NODE_LIST')
		return self
	@property
	def iter(self): return iter(self)
	@property
	def prefix(self) -> Memelang: return ''
	@property
	def suffix(self) -> Memelang: return ''
	
	def __str__(self) -> Memelang:
		return re.sub(r'\s+', ' ', self.prefix + self.opr.lex.join(map(str, self.iter)) + self.suffix)

	@property
	def kinds(self) -> List[str]:
		kinds=[]
		for i in self: kinds.extend(i.kinds)
		return kinds


# DAT [MOD DAT]
class Term(Node):
	opr: Token = TOK_SEP_TOK

TERM_ELIDE = Term(Token('VAL',ELIDE))


# TERM {OR TERM}
class Right(Node):
	opr: Token = TOK_OR
	def check(self) -> 'Right':
		if len(self) and any(not isinstance(t, Term) for t in self): raise Err('E_RIGHT_TERM')
		return self


# TERM {FUNC VAR|ALNUM}
class Left(Node): 
	opr: Token = TOK_SF
	def check(self) -> 'Left':
		if not len(self): raise Err('E_NODE_LIST')
		if not isinstance(self[0], Term): raise Err('E_LEFT_TERM')
		if len(self)>1 and any(t.kind not in {'ALNUM','VAR'} for t in self[1:]): raise Err('E_LEFT_FUNC')
		return self


# LEFT CMP RIGHT
class Axis(Node):
	opr: Token = TOK_SEP_PASS
	def check(self) -> 'Axis': 
		if len(self)!=2: raise Err('E_NODE_LIST')
		if not isinstance(self[0], Left): raise Err('E_AXIS_LEFT')
		if not isinstance(self[1], Right): raise Err('E_AXIS_RIGHT')
		return self
	@property
	def single(self) -> Token:
		if self.opr.kind == 'SEP_PASS': return Token('VAL', VAL)
		return TOK_NULL if self.opr.kind != 'EQL' or len(self[R])!=1 or len(self[R][L])!=1 else self[R][L][L]


# AXIS {SA AXIS}
class Vector(Node):
	opr: Token = TOK_SA
	mode: str = Q
	@property
	def iter(self): return reversed(self)
	@property
	def prefix(self) -> Memelang: return '' if self.mode == Q else (self.mode + SA)


# VEC {SV VEC}
class Matrix(Node):
	opr: Token = TOK_SV
	def pad(self, padding:Axis) -> None:
		max_len = 0
		for idx, vec in enumerate(self):
			if vec.mode != Q: continue
			if not max_len: max_len = len(vec)
			diff = max_len - len(vec)
			if diff>0: self[idx] += [padding] * diff
			elif diff<0: raise Err('E_FIRST_VECTOR_MUST_BE_LONGEST')



def lex(src: Memelang) -> Iterator[Token]:
	for m in MASTER_PATTERN.finditer(src):
		kind = m.lastgroup
		if kind in IGNORE_KINDS: continue
		if kind=='MISMATCH': raise Err('E_TOK')
		yield Token(kind, m.group())


def parse(src: Memelang, mode: str = Q) -> Iterator[Matrix]:
	tokens = Stream(lex(src))
	bind: List[str] = []
	mat, vec = Matrix(), Vector()
	
	while tokens.peek():

		# AXIS
		axis = Axis(Left(), Right())

		# MODE
		if tokens.peek()=='MODE':
			if vec: raise Err('E_VEC_MODE')
			mode = tokens.next().lex
			if mode not in VOCAB: raise Err('E_MODE')

		# WILD
		if tokens.peek()=='VAL': axis[L].append(Term(tokens.next()))

		# NON-WILD
		else:
			# LEFT
			if tokens.peek() in VOCAB[mode]['MOD']: axis[L].append(parse_term(Token('VAL', ELIDE), tokens, bind, mode))

			# FUNC
			while tokens.peek()=='SF':
				if not axis[L]: axis[L].append(TERM_ELIDE)
				tokens.next()
				t = tokens.next()
				if t.kind=='VAR': bind.append(t.lex)
				elif t.lex not in VOCAB[mode]['FUNC']: raise Err('E_FUNC_NAME')
				axis[L].append(t)
				
			# CMP
			if tokens.peek() in VOCAB[mode]['CMP']:
				axis.opr=tokens.next()
				if tokens.peek() not in VOCAB[mode]['DAT']: raise Err('E_CMP_DAT')

			# RIGHT
			while tokens.peek() in VOCAB[mode]['DAT']:
				if not axis[L]: axis[L].append(TERM_ELIDE)
				if axis.opr.kind=='SEP_PASS': axis.opr=Token('EQL', ELIDE)
				axis[R].append(parse_term(tokens.next(), tokens, bind, mode))
				if tokens.peek()=='OR':
					tokens.next()
					if tokens.peek() not in VOCAB[mode]['DAT']: raise Err('E_OR_TRAIL')
				if tokens.peek() == 'MODE': raise Err('E_RIGHT_MODE')

			if axis.opr.kind in VOCAB[mode]['CMP'] and not axis[R]: raise Err('E_CMP_RIGHT')

		if axis[L]:	
			axis[L], axis[R] = axis[L].check(), axis[R].check()
			vec.prepend(axis.check()) # AXES HIGH->LOW
			continue

		# VECTOR
		if tokens.peek()=='SV':
			if vec: 
				vec.mode=mode
				mat.append(vec.check())
			vec = Vector()
			tokens.next()
			bind.append(SAME)
			continue

		# MATRIX
		if tokens.peek()=='SM':
			if vec: 
				vec.mode=mode
				mat.append(vec.check())
			if mat: yield mat.check()
			mat, vec = Matrix(), Vector()
			tokens.next()
			continue

		if tokens.peek()=='SA':
			tokens.next()
			continue

		raise Err('E_TOK')

	if vec: 
		vec.mode=mode
		mat.append(vec.check())
	if mat: yield mat.check()


def parse_term (token: Token, tokens: Stream, bind: List[str], mode: str) -> Term:
	term = Term(token)
	if tokens.peek() in VOCAB[mode]['MOD']:
		term.opr=tokens.next()
		t = tokens.next()
		if t.kind not in VOCAB[mode]['DAT']: raise Err('E_TERM_DAT')
		if t.kind in {'SAME', 'VAR'} and t.lex not in bind: raise Err('E_VAR_BIND')
		term.append(t)
	return term.check()

class Meme(Node):
	opr: Token = TOK_SM
	src: Memelang

	def __init__(self, src: Memelang):
		self.src = src
		super().__init__(*parse(src))
		self.check()

	def check(self) -> 'Meme':
		for mat in self:
			if not isinstance(mat, Matrix): raise TypeError('E_TYPE_MAT')
			for vec in mat:
				if not isinstance(vec, Vector): raise TypeError('E_TYPE_VEC')
				for axis in vec:
					if not isinstance(axis, Axis): raise TypeError('E_TYPE_AXIS')
			pad_axis=Axis(Left(TERM_ELIDE), Right(Term(Token('SAME',ELIDE))))
			pad_axis.opr=Token('EQL',ELIDE)
			mat.pad(pad_axis)

		return self

	@property
	def suffix(self) -> Memelang: return SM

	def embed(self):
		for mat in self:
			for vec in mat:
				for axis in vec:
					for bucket in (axis[L][0:1], axis[R]):
						for term in bucket:
							if term.opr.kind in {'COS','L2','IP'} and len(term)==2 and term[R].kind in {'QUOT','ALNUM'}: term[R] = self.embedify(term[R])

	# OVERWRITE WITH YOUR EMBEDDING FUNCTION
	def embedify(self,tok: Token) -> Token:
		if tok.kind not in {'QUOT','ALNUM'}: raise Err('E_EMBED')
		return Token('EMB', '[0.1,0.2]')


# GENERATE RANDOM MEMELANG DATA
class Fuzz():
	@staticmethod
	def dat(kind:str) -> Memelang:
		if kind=='ALNUM': return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWMYZ') for _ in range(5))
		if kind=='QUOT': return json.dumps(''.join(random.choice(' -_+,./<>[]{}\'"!@#$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWMYZ') for _ in range(10)))
		if kind=='DEC': return str(random.uniform(-9, 9))
		if kind=='VAR': return SIGIL + Fuzz.dat('ALNUM')
		if kind=='EMB': return '[' + ','.join([str(random.uniform(0,1)) for _ in range(4)]) + ']'

	@staticmethod
	def left(mode: str) -> Memelang:
		left = ''
		if mode=='VEC': left += random.choice(['<=>','<->','<#>']) + Fuzz.dat('EMB') 
		return left

	@staticmethod
	def term(mode: str) -> Memelang:
		if mode=='NUM':
			term = Fuzz.dat('DEC')
			if random.randint(0,1): term += random.choice(['+','-','*','/']) + Fuzz.dat('DEC')
		elif mode=='STR': term = random.choice([Fuzz.dat('QUOT'), Fuzz.dat('ALNUM')])
		elif mode=='VEC': term = Fuzz.dat('EMB')
		return term

	@staticmethod
	def right(mode: str) -> Memelang:
		if mode=='STR': return OR.join(Fuzz.term(mode) for _ in range(random.randint(1,5)))
		else: return Fuzz.term(mode)

	@staticmethod
	def axis(bind: List[str]|None = None) -> Memelang:
		if not bind: bind = []

		mode: str
		data: Memelang = ''

		comp = random.choice(['=','!=','>','<','<=','>='])

		if comp in {'<','<=','>','>='}: mode = 'NUM' if random.randint(0, 3) else 'VEC'
		else: mode='STR'

		return Fuzz.left(mode) + comp + Fuzz.right(mode)

	@staticmethod
	def vector(axis_len:int = 4) -> Memelang:
		bind, vector = [], []
		for i in range(axis_len):
			if i>0: bind.append(SAME)
			vector.append(Fuzz.axis(bind))
		return SA.join(vector)

	@staticmethod
	def mat_table(col_len:int = 5) -> Memelang:
		return Fuzz.dat('ALNUM') + SA + VAL + SA + SV.join(Fuzz.dat('ALNUM') + SA + Fuzz.axis() for _ in range(col_len)) + SM



### SQL ### 

'''
1. EXAMPLE TABLE DEFINITIONS
%d roles _ id :ROL=ID; :TYP=INT; >0; rating :TYP=DEC; >0; <=5; actor :TYP=STR; movie :TYP=STR;;
%d actors _ id :ROL=ID; :TYP=INT; >0; name :TYP=STR;; age :TYP=INT; >=0; <200;;
%d movies _ id :ROL=ID; :TYP=INT; >0; description :TYP=STR; year >1800; <2100; genre scifi,drama,comedy,documentary;;

2. EXAMPLE QUERY
MEMELANG: roles _ actor "Mark Hamill",Mark; movie _; rating >4;;
SQL: SELECT t0.actor, t0.movie, t0.rating FROM roles as t0 WHERE (t0.actor = 'Mark Hamill' or t0.actor = 'Mark') AND t0.rating > 4;

3. EXAMPLE JOIN
MEMELANG: roles _ actor "Mark Hamill"; movie _; !@ @ @; actor _;;
SQL: SELECT t0.id, t0.actor, t0.movie, t1.movie, t1.actor FROM roles AS t0, roles AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.id!=t0.id AND t1.movie = t0.movie;

4. EXAMPLE TABLE JOIN WHERE ACTOR NAME = MOVIE TITLE
MEMELANG: actors _ age >21; <30; name _; roles _ title @;;
MEMELANG(2): actors _ age >21; <30; name:$n; roles _ title $n;;
SQL: SELECT t0.id, t0.name, t0.age, t1.title FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t0.age < 30 AND t1.title = t0.name;

5. EXAMPLE EMBEDDING
MEMELANG: movies _ description <=>"war":dsc>0.5; year >2005; %m lim 10; beg 100;;
MEMELANG(2): movies _ description <=>[0.1,0.2,0.3]:dsc>0.5; year >2005; %m lim 10; beg 100;;
SQL: SELECT t0.id, t0.description<=>[0.1,0.2,0.3]::VECTOR, t0.year from movies AS t0 WHERE t0.description<=>[0.1,0.2,0.3]::VECTOR>0.5 AND t0.year>2005 ORDER BY t0.description<=>[0.1,0.2,0.3]::VECTOR DESC LIMIT 10 OFFSET 100;

6. EXAMPLE AGGREGATION
MEMELANG: roles _ rating :avg; actor :grp="Mark Hamill","Carrie Fisher";;
SQL: SELECT AVG(t0.rating), t0.actor FROM roles AS t0 WHERE (t0.actor = 'Mark Hamill' OR t0.actor = 'Carrie Fisher') GROUP BY t0.actor;
'''

ANONE, ACNST, AGRP, AHAV = 0, 1, 2, 3
PH = '%s'
EPH = ' = %s'

class SQL():
	cmp2sql = {'EQL':'=','NOT':'!=','GT':'>','GE':'>=','LT':'<','LE':'<=','SMLR':'ILIKE'}
	lex: str
	alias: str
	params: List[Union[int, float, str, list]]
	agg: int|None

	def __init__(self, lex: str = '', params: List[Union[int, float, str, list]] = None, agg: int = ANONE, alias: str = ''):
		self.lex = lex
		self.agg = agg
		self.alias = alias
		self.params = [] if params is None else params

	@staticmethod
	def dat(token: Token, bind: dict) -> 'SQL':
		if token.kind in {'VAL','SAME','MSAME','VAR'}:
			if token.delide not in bind: raise Err('E_VAR_BIND')
			return bind[token.delide] if bind[token.delide] else SQL()
		return SQL(PH, [token.dat], ACNST)

	@staticmethod
	def term(term: Term, bind: dict) -> 'SQL':
		if not term: return SQL()
		dats = [SQL.dat(t, bind) for t in term]
		return SQL(term.opr.lex.join(dat.lex for dat in dats), [p for dat in dats for p in dat.params])

	@staticmethod
	def single(axis: Axis, bind: dict) -> 'SQL':
		return SQL() if axis.single.dat is None else SQL.dat(axis.single, bind)

	@staticmethod
	def select(axis: Axis, bind: dict, alias: str = '') -> 'SQL':
		agg_func = {'cnt':'COUNT(1)','sum': 'SUM', 'avg': 'AVG', 'min': 'MIN', 'max': 'MAX'}
		left = SQL.term(axis[L][L], bind)
		for t in axis[L][R:]:
			if t.lex in agg_func:
				if left.agg == AHAV: raise Err('E_DBL_AGG')
				left.agg = AHAV
				left.lex = agg_func[t.lex] + ('' if '(1)' in agg_func[t.lex] else '(' + left.lex + ')')
			elif t.lex=='grp': left.agg = AGRP
			# TO DO: FLAG CONFLICTS
		if alias: left.alias=alias
		return left
		
	@staticmethod
	def where(axis: Axis, bind: dict) -> 'SQL':
		if axis.opr.kind=='SEP_PASS': return SQL()
		sym = SQL.cmp2sql[axis.opr.kind]
		lp, rp, right, ts = '', '', '', []
		params = []

		if len(axis[R]) > 1:
			lp, rp = '(', ')'
			right = ' AND ' if axis.opr.kind=='NOT' else ' OR '

		select = SQL.select(axis, bind)
		params.extend(select.params)
		for t in axis[R]:
			where = SQL.term(t, bind)
			if sym in ('LIKE','ILIKE'): where.lex = where.lex.replace(PH, "CONCAT('%', %s, '%')", 1)
			ts.append(f"{select.lex} {sym} {where.lex}")
			params.extend(where.params)

		return SQL(lp + right.join(ts) + rp, params, select.agg)

	@property
	def holder(self) -> str:
		return self.lex + ( '' if not self.alias else f' AS "{self.alias}"')

	@property
	def param(self):
		return None if self.lex!=PH or len(self.params)!=1 else self.params[0]

	# NOT DB SAFE - FOR DEBUGGING ONLY
	def __str__(self) -> str:
		sql = self.lex
		if self.alias: sql += f' AS "{self.alias}"'
		for p in self.params:
			if isinstance(p, str): v = "'" + p.replace("'", "''") + "'"
			elif isinstance(p, list): v = json.dumps(p, separators=(',',':')) + '::VECTOR'
			elif p is None: v = 'NULL'
			else: v = str(p)
			sql = sql.replace(PH, v, 1)
		return sql

	def __eq__(self, other): return isinstance(other, SQL) and str(self)==str(other)

# TRANSLATE TO POSTGRES
class MemePGSQL(Meme):
	def select(self) -> List[SQL]:
		self.embed()
		tab_idx: int = 0
		sql: List[SQL] = []
		axes = ('val','col','row','tab')

		for mat in self:
			sel_all, tab_alias, pri_col = False, None, None
			froms, wheres, selects, ords, groups, havings, bind = [], [], [], [], [], [], {}
			prev = {name:None for name in axes}
			config = {M: {'val':0,'col':1,'row':2,'tab':3,'lim':0,'beg':0,'pri':'id'}}

			for vec in mat:

				# META VECTORS
				# ; %m lim 10; beg 100
				if vec.mode==M:
					if len(vec)!=2: raise Err('E_X_LEN')
					key, val = vec[1].single, vec[0].single
					if key.lex in config[M] and isinstance(val.dat, type(config[M][key.lex])): config[M][key.lex] = val.dat
					else: raise Err('E_MODE_KEY')
					continue

				# QUERY VECTORS
				# tab row col term:func:func>term,term;
				if vec.mode!=Q: continue
				if len(vec)<4: raise Err('E_Q_LEN')

				curr = {name: None for name in axes}
				
				# TAB
				bind[VAL],bind[SAME]=None, prev['tab']
				curr['tab'] = SQL.single(vec[config[M]['tab']], bind)

				# ROW
				bind[SAME],bind[VAL] = None, prev['row']
				curr['row'] = SQL.single(vec[config[M]['row']], bind)

				# JOIN
				if prev['tab']!=curr['tab'] or not curr['row'].lex or prev['row']!=curr['row']:
					if curr['tab'].param is None:
						print(vec)
						raise Err('E_TBL_NAME')
					tab_alias = f't{tab_idx}'
					tab_idx += 1
					#selects.append(curr['tab'])
					froms.append(SQL(curr['tab'].param, [], ACNST, tab_alias))

					bind[SAME]=prev['row']
					curr['row']=bind[VAL]=SQL(f"{tab_alias}.{config[M]['pri']}")
					selects.append(SQL.select(vec[config[M]['row']], bind))
					wheres.append(SQL.where(vec[config[M]['row']], bind))

				# COL
				if vec[config[M]['col']].single.kind=='VAL':
					sel_all=True
					continue

				bind[SAME], bind[VAL] = prev['col'], None
				curr['col'] = SQL.single(vec[config[M]['col']], bind)
				if curr['col'].param is None: raise Err('E_COL_NAME')
				#selects.append(curr['col'])

				# VAL
				bind[SAME]=prev['val']
				bind[VAL]=SQL(tab_alias + '.' + curr['col'].params[0])
				curr['val'] = SQL.single(vec[config[M]['val']], bind)
				wheres.append(SQL.where(vec[config[M]['val']], bind))

				select = SQL.select(vec[config[M]['val']], bind, f"{tab_alias}.{curr['col'].param}")
				selects.append(select)
				#selects.append(SQL("';'", [], ACNST))

				# AGG/SORT
				funcs = set(t.lex for t in vec[config[M]['val']][L][R:])
				if 'grp' in funcs: groups.append(select)
				if 'asc' in funcs: ords.append(SQL(select.lex+' ASC', select.params, ANONE))
				elif 'dsc' in funcs: ords.append(SQL(select.lex+' DESC', select.params, ANONE))

				# BIND VARS			
				for name in axes:
					for t in vec[config[M][name]][L][R:]:
						if t.kind=='VAR': bind[t.lex]=curr[name]

				prev = curr.copy()

			if groups: 
				#if sel_all: raise Err('E_SELALL_GRP')
				selects = [s for s in selects if s.agg>ANONE]
			elif sel_all: selects.append(SQL(f'{tab_alias}.*', [], ANONE))

			SQLPARTS=[
				['SELECT', ', ', selects, True],
				['FROM', ', ', froms, True],
				['WHERE', ' AND ', [p for p in wheres if p.agg<AHAV], False],
				['GROUP BY', ', ', groups, False],
				['HAVING', ' AND ', [p for p in wheres if p.agg==AHAV], False],
				['ORDER BY', ', ', ords, False]
			]

			sqlstr,space,params='','',[]
			for keyword, sep, items, usealias in SQLPARTS:
				if not items: continue
				sqlstr+=space+keyword+' '+sep.join([(s.holder if usealias else s.lex)  for s in items if s.lex])
				params.extend(p for s in items for p in s.params if p is not None)
				space = ' '

			if config[M]['lim']: sqlstr += f" LIMIT {config[M]['lim']}"
			if config[M]['beg']: sqlstr += f" OFFSET {config[M]['beg']}"

			sql.append(SQL(sqlstr, params))

		return sql


### CLI ###

if __name__ == "__main__":
	if len(sys.argv)==2:
		meme = MemePGSQL(sys.argv[1])
		print(str(meme))
		print(meme.select())
	elif len(sys.argv)==3 and sys.argv[1]=='file':
		with open(sys.argv[2], 'r', encoding='utf-8') as f: data = json.load(f)
		for idx, example in enumerate(data['examples']):
			print(f"{idx}. {example['input']}")
			print(example['output'])
			meme = MemePGSQL(example['output'])
			print(str(meme))
			print(str(meme.select()[0]))
			print()
		print('SUCCESS')
	else: raise Err('E_ARG')
