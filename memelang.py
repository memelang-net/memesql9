'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs
MEMELANG USES AXES ORDERED HIGH TO LOW
ALWAYS WHITESPACE MEANS "NEW AXIS"
NEVER SPACE AROUND OPERATOR
NEVER SPACE BETWEEN COMPARATOR/COMMA/FUNC AND VALUES
'''

MEMELANG_VER = 9.18

import random, re, json
from typing import List, Iterator, Iterable, Dict, Tuple, Union

Memelang = str
Err = SyntaxError

ELIDE, SIGIL, VAL, MSAME, SAME,  META, EOF =  '', '$', '_', '^', '@',  '%', None
SA, SV, SM, SF, OR = ' ', ';', ';;', ':', ','
L, R = 0, 1
Q, F = META+'q', META+'f'


TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOT',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS "John \"Jack\" Kennedy"
	('MTBL',		r'-*\|'),
	('EMB',			r'\[(?:-?\d+(?:\.\d+)?)(?:,-?\d+(?:\.\d+)?)*\]'), # JSON ARRAY OF FLOATS [0.1,0.2]
	('POW',			r'\*\*'),
	('MUL',			r'\*'),
	('ADD',			r'\+'),
	('DIV',			r'\/'),
	('META',		re.escape(META) + r'[a-z]+'),
	('MOD',			r'\%'),
	#('TSQ',		r'@@'),
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
	('ALNUM',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERICS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('SUB',			r'\-'), # AFTER INT/FLOAT
	('SF',			re.escape(SF)),
	('SM',			re.escape(SM)),
	('SV',			re.escape(SV)),
	('OR',			re.escape(OR)),
	('SA',			r'\s+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

CMP_KINDS = {'EQL':{'STR','NUM','DATA'},'NOT':{'STR','NUM','DATA'},'GT':{'NUM'},'GE':{'NUM'},'LT':{'NUM'},'LE':{'NUM'},'SMLR':{'STR'}}
MOD_KINDS = {'MUL':{'NUM'},'ADD':{'NUM'},'SUB':{'NUM'},'DIV':{'NUM'},'MOD':{'NUM'},'POW':{'NUM'},'L2':{'EMB'},'IP':{'EMB'},'COS':{'EMB'}} #,'TSQ':{'TSQ'}
DAT_KINDS = {'ALNUM','QUOT','INT','FLOAT','VAR','SAME','MSAME','VAL','EMB','META'}
IGNORE_KINDS = {'COMMENT','MTBL'}
FUNC_KINDS = {'ALNUM','VAR'}

EBNF = '''
TERM ::= DAT [MOD DAT]
LEFT ::= TERM {FUNC VAR|ALNUM}
RIGHT ::= {TERM} {OR TERM}
AXIS ::= LEFT [CMP RIGHT]
VEC ::= AXIS {SA AXIS}
MAT ::= VEC {SV VEC}
MEME ::= MAT {SM MAT}
'''

class Token():
	kind: str
	kinds: List[str]
	lex: str
	dat: Union[str, float, int, list]
	def __init__(self, kind: str, lex: str):
		self.kind = kind
		self.kinds = [kind]
		self.lex = lex
		if kind=='QUOT': 	self.dat = json.loads(lex)
		elif kind=='EMB': 	self.dat = json.loads(lex)
		elif kind=='FLOAT': self.dat = float(lex)
		elif kind=='INT':	self.dat = int(lex)
		elif kind=='NULL':	self.dat = None
		else: 				self.dat = lex

	def dump(self) -> Union[str, float, int, list]: return self.dat
	def __str__(self) -> Memelang: return self.lex
	def __eq__(self, other): return isinstance(other, Token) and self.kind==other.kind and self.lex==other.lex


TOK_NULL = Token('NULL', '')
TOK_EQL = Token('EQL', ELIDE)
TOK_NOT = Token('NOT', '!')
TOK_SA = Token('SA', SA)
TOK_SV = Token('SV', SV)
TOK_SM = Token('SM', SM)
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
			elif diff<0: raise Err('E_FIRST_VECTOR_MUST_BE_LONGEST')

	def dump(self) -> List: return [self.opr.dump(), [i.dump() for i in self]]
	def check(self) -> 'Node': 
		if len(self)==0: raise Err('E_NODE_LIST')
		return self
	def __str__(self) -> Memelang: return self.opr.lex.join(map(str, self))

	@property
	def kinds(self) -> List[str]:
		kinds=[]
		for i in self: kinds.extend(i.kinds)
		return kinds


# DAT [MOD DAT]
class Term(Node):
	opr: Token = TOK_SEP_TOK
	def check(self) -> 'Term':
		if not len(self): raise Err('E_NO_LIST')
		if any(not isinstance(t, Token) or t.kind not in DAT_KINDS for t in self): raise Err('E_TERM_DAT')
		return self

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
		if not len(self): raise Err('E_NO_LIST')
		if not isinstance(self[0], Term): raise Err('E_L_TERM')
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
		return TOK_NULL if self.opr.kind!='EQL' or len(self[R])!=1 or len(self[R][L])!=1 else self[R][L][L]


# AXIS {SA AXIS}
class Vector(Node):
	opr: Token = TOK_SA
	def __str__(self) -> Memelang: return self.opr.lex.join(map(str, reversed(self))) # AXES HIGH->LOW


# VEC {SV VEC}
class Matrix(Node):
	opr: Token = TOK_SV


def lex(src: Memelang) -> Iterator[Token]:
	for m in MASTER_PATTERN.finditer(src):
		kind = m.lastgroup
		if kind in IGNORE_KINDS: continue
		if kind=='MISMATCH': raise Err('E_TOK')
		yield Token(kind, m.group())


def parse(src: Memelang) -> Iterator[Matrix]:
	tokens = Stream(lex(src))
	bind: List[str] = []
	mat, vec = Matrix(), Vector()

	while tokens.peek():

		# AXIS
		axis = Axis(Left(), Right())

		# WILD
		if tokens.peek()=='VAL': axis[L].append(Term(tokens.next()))

		# NON-WILD
		else:
			# LEFT
			if tokens.peek() in MOD_KINDS: axis[L].append(parse_term(Token('VAL', ELIDE), tokens, bind))

			# FUNC
			while tokens.peek()=='SF':
				if not axis[L]: axis[L].append(TERM_ELIDE)
				tokens.next()
				t = tokens.next()
				if t.kind not in FUNC_KINDS: raise Err('E_FUNC_KIND')
				if t.kind=='VAR': bind.append(t.lex)
				axis[L].append(t)
				
			# CMP
			if tokens.peek() in CMP_KINDS:
				axis.opr=tokens.next()
				if tokens.peek() not in DAT_KINDS: raise Err('E_CMP_DAT')

			# RIGHT
			while tokens.peek() in DAT_KINDS:
				if not axis[L]: axis[L].append(TERM_ELIDE)
				if axis.opr.kind=='SEP_PASS': axis.opr=Token('EQL', ELIDE)
				axis[R].append(parse_term(tokens.next(), tokens, bind))
				if tokens.peek()=='OR':
					tokens.next()
					if tokens.peek() not in DAT_KINDS: raise Err('E_OR_TRAIL')

			if axis.opr.kind in CMP_KINDS and not axis[R]: raise Err('E_CMP_RIGHT')

		if axis[L]:	
			axis[L], axis[R] = axis[L].check(), axis[R].check()
			vec.prepend(axis.check()) # AXES HIGH->LOW
			continue

		# VEC
		if tokens.peek()=='SV':
			if vec: mat.append(vec.check())
			vec = Vector()
			tokens.next()
			bind.append(SAME)
			continue

		# MAT
		if tokens.peek()=='SM':
			if vec: mat.append(vec.check())
			if mat: yield mat.check()
			mat, vec = Matrix(), Vector()
			tokens.next()
			continue

		if tokens.peek()=='SA':
			tokens.next()
			continue

		raise Err('E_TOK')

	if vec: mat.append(vec.check())
	if mat: yield mat.check()


def parse_term (token: Token, tokens: Stream, bind: List[str]) -> Term:
	term = Term(token)
	if tokens.peek() in MOD_KINDS:
		term.opr=tokens.next()
		t = tokens.next()
		if t.kind not in DAT_KINDS: raise Err('E_EXPR_DAT')
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
		for mat_idx, mat in enumerate(self):
			if not isinstance(mat, Matrix): raise TypeError('E_TYPE_MAT')
			for vec_idx, vec in enumerate(mat):
				if not isinstance(vec, Vector): raise TypeError('E_TYPE_VEC')
				for axis_idx, axis in enumerate(vec):
					if not isinstance(axis, Axis): raise TypeError('E_TYPE_AXIS')
					# DO VAR BIND HERE
			self[mat_idx].pad(Axis(
					Left(TERM_ELIDE).check(),
					Right(Term(Token('SAME',SAME)).check()).check(),
					opr=Token('EQL',ELIDE)
				).check()
			)

		return self


# GENERATE RANDOM MEMELANG DATA
class Fuzz():
	@staticmethod
	def dat(kind:str) -> Memelang:
		if kind=='ALNUM': return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(5))
		if kind=='QUOT': return json.dumps(''.join(random.choice(' -_+,./<>[]{}\'"!@#$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(10)))
		if kind=='FLOAT': return str(random.uniform(-9, 9))
		if kind=='VAR': return SIGIL + Fuzz.dat('ALNUM')
		if kind=='EMB': return '[' + ','.join([str(random.uniform(0,1)) for _ in range(4)]) + ']'

	@staticmethod
	def lterm(mode: str) -> Memelang:
		term = ''
		if mode=='VEC': term += random.choice(['<=>','<->','<#>']) + Fuzz.dat('EMB') 
		return term

	@staticmethod
	def rterm(mode: str) -> Memelang:
		if mode=='NUM':
			term = Fuzz.dat('FLOAT')
			if random.randint(0,1): term += random.choice(['+','-','*','/']) + Fuzz.dat('FLOAT')
		elif mode=='STR': term = random.choice([Fuzz.dat('QUOT'), Fuzz.dat('ALNUM')])
		elif mode=='VEC': term = Fuzz.dat('EMB')
		return term

	@staticmethod
	def junc(mode: str) -> Memelang:
		if mode=='STR': return OR.join(Fuzz.rterm(mode) for _ in range(random.randint(1,5)))
		else: return Fuzz.rterm(mode)

	@staticmethod
	def limit(bind: List[str]|None = None) -> Memelang:
		if not bind: bind = []

		mode: str
		data: Memelang = ''

		comp = random.choice(['=','!=','>','<','<=','>='])

		if comp in {'<','<=','>','>='}: mode = 'NUM' if random.randint(0, 3) else 'VEC'
		else: mode='STR'

		return Fuzz.lterm(mode) + comp + Fuzz.junc(mode)

	@staticmethod
	def vector(limit_len:int = 4) -> Memelang:
		bind, vector = [], []
		for i in range(limit_len):
			if i>0: bind.append(SAME)
			vector.append(Fuzz.limit(bind))
		return SA.join(vector)

	@staticmethod
	def mat_table(col_len:int = 5) -> Memelang:
		return Fuzz.dat('ALNUM') + SA + VAL + SA + SV.join(Fuzz.dat('ALNUM') + SA + Fuzz.limit() for _ in range(col_len)) + SM



### SQL ### 

'''
1. EXAMPLE QUERY
MEMELANG: roles _ actor "Mark Hamill",Mark; movie _; rating >4;;
SQL: SELECT t0.actor, t0.movie, t0.rating FROM roles as t0 WHERE (t0.actor = 'Mark Hamill' or t0.actor = 'Mark') AND t0.rating > 4;

2. EXAMPLE JOIN
MEMELANG: roles _ actor "Mark Hamill"; movie _; !@ @ @; actor _;;
SQL: SELECT t0.id, t0.actor, t0.movie, t1.movie, t1.actor FROM roles AS t0, roles AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.id!=t0.id AND t1.movie = t0.movie;

3. EXAMPLE TABLE JOIN WHERE ACTOR NAME = MOVIE TITLE
MEMELANG: actors _ age >21; name _; roles _ title @;;
MEMELANG(2): actors _ age >21; name:$n; roles _ title $n;;
SQL: SELECT t0.id, t0.name, t0.age, t1.title FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t1.title = t0.name;

4. EXAMPLE EMBEDDING
MEMELANG: movies _ description <=>[0.1,0.2,0.3]:dsc>0.5; year >2005; %q lim 10; beg 100;;
SQL: SELECT t0.id, t0.description<=>[0.1,0.2,0.3], t0.year from movies AS t0 WHERE t0.description<=>[0.1,0.2,0.3]>0.5 AND t0.year>2005 ORDER BY t0.description<=>[0.1,0.2,0.3] DESC LIMIT 10 OFFSET 100;

5. EXAMPLE AGGREGATION
MEMELANG: roles _ rating :avg; actor :grp="Mark Hamill","Carrie Fisher";;
SQL: SELECT AVG(t0.rating), t0.actor FROM roles AS t0 WHERE (t0.actor = 'Mark Hamill' OR t0.actor = 'Carrie Fisher') GROUP BY t0.actor;

'''

SQL = str
Param = int|float|str|list
Agg = int
ANONE, ACNST, AGRP, AHAV = 0, 1, 2, 3

class SQLUtil():
	cmp2sql = {'EQL':'=','NOT':'!=','GT':'>','GE':'>=','LT':'<','LE':'<=','SMLR':'ILIKE'}
	func_keys = {"grp","asc","dsc","sum","avg","min","max"}

	@staticmethod
	def holder(token: Token, bind: dict) -> SQL:
		if token.kind=='DBCOL': return token.dat
		elif token.kind=='VAL': return SQLUtil.holder(bind[VAL], bind)
		elif token.kind=='SAME': return SQLUtil.holder(bind[SAME], bind)
		elif token.kind=='VAR':
			if token.lex not in bind: raise Err('E_VAR_BIND')
			return SQLUtil.holder(bind[token.lex], bind)
		return '%s'

	@staticmethod
	def param(token: Token, bind: dict) -> None|Param:
		if token.kind=='DBCOL': return None
		elif token.kind=='VAL': return SQLUtil.param(bind[VAL], bind)
		elif token.kind=='SAME': return SQLUtil.param(bind[SAME], bind)
		elif token.kind=='VAR':
			if token.lex not in bind: raise Err('E_VAR_BIND')
			return SQLUtil.param(bind[token.lex], bind)
		return token.dat

	@staticmethod
	def term(term: Term, bind: dict) -> Tuple[SQL, List[None|Param]]:
		sqlterm = SQLUtil.holder(term[0], bind)
		sqlparams = [SQLUtil.param(term[0], bind)]
		if term.opr.kind!='SEP_TOK':
			sqlterm += term.opr.lex + SQLUtil.holder(term[1], bind)
			sqlparams.append(SQLUtil.param(term[1], bind))

		return sqlterm, sqlparams

	@staticmethod
	def select(axis: Axis, bind: dict) -> Tuple[SQL, List[None|Param], Agg]:
		agg_func = {'sum': 'SUM', 'avg': 'AVG', 'min': 'MIN', 'max': 'MAX'}
		agg = ANONE
		sqlterm, sqlparams = SQLUtil.term(axis[L][L], bind)
		for t in axis[L][R:]:
			if t.lex in agg_func:
				if agg: raise Err('E_DBL_AGG')
				agg = AHAV
				sqlterm = agg_func[t.lex] + '(' + sqlterm + ')'
			elif t.lex=='grp': agg = AGRP
			# TO DO: FLAG CONFLICTS
		return sqlterm, sqlparams, agg
		
	@staticmethod
	def where(axis: Axis, bind: dict) -> Tuple[SQL, List[None|Param], Agg]:
		if axis.opr.kind=='SEP_PASS': return '', [], ANONE
		sym = SQLUtil.cmp2sql[axis.opr.kind]
		lp, rp, junc, ts = '', '', '', []

		if len(axis[R]) > 1:
			lp, rp = '(', ')'
			junc = ' AND ' if axis.opr.kind=='NOT' else ' OR '

		leftsql, params, agg = SQLUtil.select(axis, bind)
		for t in axis[R]:
			sql, subparams = SQLUtil.term(t, bind)
			if sym in ('LIKE','ILIKE'): sql = sql.replace('%s', "CONCAT('%', %s, '%')", 1)
			ts.append(f"{leftsql} {sym} {sql}")
			params.extend(subparams)

		return lp + junc.join(ts) + rp, params, agg

	@staticmethod
	def deref(axis: Axis, bind: dict) -> Tuple[bool, Token]:
		if axis.single.kind=='SAME': return True, bind.get(SAME)
		return (bind.get(SAME) is not None and axis.single==bind.get(SAME)), axis.single


# TRANSLATE TO POSTGRES
class MemePGSQL(Meme):
	def select(self) -> List[Tuple[SQL, List[Param]]]:
		tab_idx: int = 0
		sql: List[Tuple[SQL, List[Param]]] = []
		axes = ('val','col','row','tab')
		
		for mat in self:
			selectall = False
			tab_alias = None
			limstr = ''
			froms, wheres, selects, ords, groups, havings, bind = [], [], [], [], [], [], {}
			prev = {name:None for name in axes}
			config = {Q: {'val':0,'col':1,'row':2,'tab':3,'lim':0,'beg':0,'pri':'id'}}

			# CONFIG VECTORS
			# ; %q lim 10; beg 100
			mode = F
			for vec in mat:
				mode = next((a.single.lex for a in vec if a.single.kind=='META'), mode)
				if mode!=Q: continue
				if len(vec)<2: raise Err('E_Q_LEN')
				key, val = vec[1].single, vec[0].single
				if key.lex in config[mode] and isinstance(val.dat, type(config[mode][key.lex])): config[mode][key.lex] = val.dat
				else: raise Err('E_META_KEY')					

			# FILTER VECTORS
			# tab row col term:meta>term,term;
			mode = F
			for vec in mat:
				mode = next((a.single.lex for a in vec if a.single.kind=='META'), mode)
				if mode!=F: continue
				if len(vec)<4: raise Err('E_SQL_VEC_LEN')

				curr = {name: None for name in axes}
				same = {name: None for name in axes}
				for name in axes: same[name], curr[name] = SQLUtil.deref(vec[config[Q][name]], {SAME: prev[name]})

				valaxis = vec[config[Q]['val']]

				# JOIN
				if not same['tab'] or not same['row']:

					if selectall: selects.append((f'{tab_alias}.*', [], ANONE))
					selectall = False

					# TBL
					if not curr['tab'] or curr['tab'].kind!='ALNUM': raise Err('E_TBL_ALNUM')
					tab_alias = f't{tab_idx}'
					froms.append(f"{curr['tab']} AS {tab_alias}")
					tab_idx += 1
					pricol = f"{tab_alias}.{config[Q]['pri']}"

					# ROW
					bind[SAME]=prev['row'] if prev['row'] is not None else None
					curr['row']=bind[VAL]=Token('DBCOL', pricol)
					where, param, _ = SQLUtil.where(vec[config[Q]['row']], bind)
					if where: wheres.append((where, param, ANONE))

					selects.extend([(f"'{curr['tab'].lex}' AS _a3", [], ACNST), (f"{pricol} AS _a2", [], ANONE)])

				# COL
				if not curr['col']: raise Err('E_COL')
				elif curr['col'].kind=='VAL' and vec[config[Q]['col']].opr.kind=='SEP_PASS':
					selectall=True
					continue
				elif curr['col'].kind!='ALNUM': raise Err('E_COL_ALNUM')

				col_name = curr['col'].dat
				col_alias = f"{tab_alias}.{col_name}"

				# VAL
				if prev['val']: bind[SAME]=prev['val']
				curr['val']=bind[VAL]=Token('DBCOL', col_alias)

				select = SQLUtil.select(valaxis, bind)
				selects.append(select)

				if any(t.lex=='grp' for t in valaxis[L][R:]): groups.append(select)
				if any(t.lex=='asc' for t in valaxis[L][R:]): ords.append((select[0]+' ASC', select[1], select[2]))
				elif any(t.lex=='dsc' for t in valaxis[L][R:]): ords.append((select[0]+' DESC', select[1], select[2]))

				where = SQLUtil.where(valaxis, bind)
				if where[0]:
					if where[2]==AHAV: havings.append(where)
					else: wheres.append(where)
			
				for name in axes:
					for t in vec[config[Q][name]][L][R:]:
						if t.kind=='VAR': bind[t.lex]=curr[name]

				prev = curr.copy()

			if groups: 
				if selectall: raise Err('E_SLCTALL_AGRP')
				selects = [s for s in selects if s[2]>ANONE]
			elif selectall: selects.append((f'{tab_alias}.*', [], ANONE))

			selectstr = 'SELECT ' + ', '.join([s[0] for s in selects if s[0]])
			fromstr = ' FROM ' + ', '.join(froms)
			wherestr = '' if not wheres else ' WHERE ' + ' AND '.join([s[0] for s in wheres if s[0]])
			groupstr = '' if not groups else ' GROUP BY ' + ', '.join([s[0] for s in groups if s[0]])
			havingstr = '' if not havings else ' HAVING ' + ' AND '.join([s[0] for s in havings if s[0]])
			ordstr = '' if not ords else ' ORDER BY ' + ', '.join([s[0] for s in ords if s[0]])
			if config[Q]['lim']: limstr += f" LIMIT {config[Q]['lim']}"
			if config[Q]['beg']: limstr += f" OFFSET {config[Q]['beg']}"
			params = [p for s in selects+wheres+groups+havings+ords for p in s[1] if p is not None]
			sql.append((selectstr + fromstr + wherestr + groupstr + havingstr + ordstr + limstr, params))

		return sql
