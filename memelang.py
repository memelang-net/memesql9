'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs
MEMELANG USES AXES
AXES ORDERED HIGH TO LOW
ALWAYS WHITESPACE MEANS "NEW AXIS"
NEVER SPACE AROUND OPERATOR
NEVER SPACE BETWEEN COMPARATOR/COMMA/FUNC AND VALUES
'''

MEMELANG_VER = 9.17

import random, re, json
from typing import List, Iterator, Iterable, Dict, Tuple, Union

Memelang = str

ELIDE, SIGIL, VAL, MSAME, VSAME, SEP_FUNC, META, EOF =  '', '$', '_', '^', '@', ':', '%', None
SEP_AXIS, SEP_VCTR, SEP_MTRX, SEP_OR = ' ', ';', ';;', ','
L, R = 0, 1
AXIS_QI, AXIS_QK, AXIS_QV = 2, 1, 0


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
	('MSAME',		re.escape(MSAME)),		# REFERENCES (MTRX-1, VCTR=-1, LIMIT)
	('VSAME',		re.escape(VSAME)),		# REFERENCES (MTRX,   VCTR-1,  LIMIT)
	('VAR',			re.escape(SIGIL) + r'[A-Za-z0-9_]+'),
	('ALNUM',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERICS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('SUB',			r'\-'), # AFTER INT/FLOAT
	('SEP_FUNC',	re.escape(SEP_FUNC)),
	('SEP_MTRX',	re.escape(SEP_MTRX)),
	('SEP_VCTR',	re.escape(SEP_VCTR)),
	('SEP_OR',		re.escape(SEP_OR)),
	('SEP_AXIS',	r'\s+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

CMP_KINDS = {'EQL':{'STR','NUM','DATA'},'NOT':{'STR','NUM','DATA'},'GT':{'NUM'},'GE':{'NUM'},'LT':{'NUM'},'LE':{'NUM'},'SMLR':{'STR'}}
MOD_KINDS = {'MUL':{'NUM'},'ADD':{'NUM'},'SUB':{'NUM'},'DIV':{'NUM'},'MOD':{'NUM'},'POW':{'NUM'},'L2':{'EMB'},'IP':{'EMB'},'COS':{'EMB'}} #,'TSQ':{'TSQ'}
DAT_KINDS = {'ALNUM','QUOT','INT','FLOAT','VAR','VSAME','MSAME','VAL','EMB','META'}
IGNORE_KINDS = {'COMMENT','MTBL'}
FUNC_KINDS = {'ALNUM','VAR'}

EBNF = '''
TERM ::= DAT [MOD DAT]
LEFT ::= TERM {FUNC VAR|ALNUM}
RIGHT ::= {TERM} {SEP_OR TERM}
AXIS ::= LEFT [CMP RIGHT]
VCTR ::= AXIS {SEP_AXIS AXIS}
MTRX ::= VCTR {SEP_VCTR VCTR}
MEME ::= MTRX {SEP_MTRX MTRX}
'''

class Token():
	kind: str
	kinds: List[str]
	lexeme: str
	datum: Union[str, float, int, list]
	def __init__(self, kind: str, lexeme: str):
		self.kind = kind
		self.kinds = [kind]
		self.lexeme = lexeme
		if kind == 'QUOT': 		self.datum = json.loads(lexeme)
		elif kind == 'EMB': 	self.datum = json.loads(lexeme)
		elif kind == 'FLOAT': 	self.datum = float(lexeme)
		elif kind == 'INT':		self.datum = int(lexeme)
		elif kind == 'NULL':	self.datum = None
		else: 					self.datum = lexeme

	def dump(self) -> Union[str, float, int, list]: return self.datum
	def __str__(self) -> Memelang: return self.lexeme
	def __eq__(self, other): return isinstance(other, Token) and self.kind == other.kind and self.lexeme == other.lexeme


TOK_NULL = Token('NULL', '')
TOK_EQL = Token('EQL', ELIDE)
TOK_NOT = Token('NOT', '!')
TOK_SEP_AXIS = Token('SEP_AXIS', SEP_AXIS)
TOK_SEP_VCTR = Token('SEP_VCTR', SEP_VCTR)
TOK_SEP_MTRX = Token('SEP_MTRX', SEP_MTRX)
TOK_SEP_OR = Token('SEP_OR', SEP_OR)
TOK_SEP_FUNC = Token('SEP_FUNC', SEP_FUNC)
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
			if val is EOF: raise SyntaxError('E_EOF')
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
			elif diff<0: raise SyntaxError('E_FIRST_VECTOR_MUST_BE_LONGEST')

	def dump(self) -> List: return [self.opr.dump(), [item.dump() for item in self]]
	def check(self) -> 'Node': 
		if len(self)==0: raise SyntaxError('E_NODE_LIST')
		return self
	def __str__(self) -> Memelang: return self.opr.lexeme.join(map(str, self))

	@property
	def kinds(self) -> List[str]:
		kinds=[]
		for n in self: kinds.extend(n.kinds)
		return kinds


# DAT [MOD DAT]
class Term(Node):
	opr: Token = TOK_SEP_TOK
	def check(self) -> 'Term':
		if not len(self): raise SyntaxError('E_NO_LIST')
		if any(not isinstance(t, Token) or t.kind not in DAT_KINDS for t in self): raise SyntaxError('E_TERM_DAT')
		return self

TERM_ELIDE = Term(Token('VAL',ELIDE))


# TERM {OR TERM}
class Right(Node):
	opr: Token = TOK_SEP_OR
	def check(self) -> 'Right':
		if len(self) and any(not isinstance(t, Term) for t in self): raise SyntaxError('E_RIGHT_TERM')
		return self


# TERM {FUNC VAR|ALNUM}
class Left(Node): 
	opr: Token = TOK_SEP_FUNC
	def check(self) -> 'Left':
		if not len(self): raise SyntaxError('E_NO_LIST')
		if not isinstance(self[0], Term): raise SyntaxError('E_L_TERM')
		if len(self)>1 and any(t.kind not in {'ALNUM','VAR'} for t in self[1:]): raise SyntaxError('E_LEFT_FUNC')
		return self


# LEFT CMP RIGHT
class Axis(Node):
	opr: Token = TOK_SEP_PASS
	def check(self) -> 'Axis': 
		if len(self)!=2: raise SyntaxError('E_NODE_LIST')
		if not isinstance(self[0], Left): raise SyntaxError('E_AXIS_LEFT')
		if not isinstance(self[1], Right): raise SyntaxError('E_AXIS_RIGHT')
		return self
	@property
	def singleton(self) -> Token:
		return TOK_NULL if self.opr.kind != 'EQL' or len(self[R])!=1 or len(self[R][L])!=1 else self[R][L][L]


# AXIS {SEP_AXIS AXIS}
class Vector(Node):
	opr: Token = TOK_SEP_AXIS
	def __str__(self) -> Memelang: return self.opr.lexeme.join(map(str, reversed(self)))


# VCTR {SEP_VCTR VCTR}
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
	bindings: List[str] = []
	mtrx, vctr = Matrix(), Vector()

	while tokens.peek():

		# AXIS
		axis = Axis(Left(), Right())

		# WILD
		if tokens.peek() == 'VAL': axis[L].append(Term(tokens.next()))

		# NON-WILD
		else:
			# LEFT
			if tokens.peek() in MOD_KINDS: axis[L].append(parse_term(Token('VAL', ELIDE), tokens, bindings))

			# FUNC
			while tokens.peek() == 'SEP_FUNC':
				if not axis[L]: axis[L].append(TERM_ELIDE)
				tokens.next()
				t = tokens.next()
				if t.kind not in FUNC_KINDS: raise SyntaxError('E_FUNC_KIND')
				if t.kind == 'VAR': bindings.append(t.lexeme)
				axis[L].append(t)
				
			# CMP
			if tokens.peek() in CMP_KINDS:
				axis.opr=tokens.next()
				if tokens.peek() not in DAT_KINDS: raise SyntaxError('E_CMP_DAT')

			# RIGHT
			while tokens.peek() in DAT_KINDS:
				if not axis[L]: axis[L].append(TERM_ELIDE)
				if axis.opr.kind == 'SEP_PASS': axis.opr=Token('EQL', ELIDE)
				axis[R].append(parse_term(tokens.next(), tokens, bindings))
				if tokens.peek() == 'SEP_OR':
					tokens.next()
					if tokens.peek() not in DAT_KINDS: raise SyntaxError('E_OR_TRAIL')

			if axis.opr.kind in CMP_KINDS and not axis[R]: raise SyntaxError('E_CMP_RIGHT')

		if axis[L]:	
			axis[L], axis[R] = axis[L].check(), axis[R].check()
			vctr.prepend(axis.check())
			continue

		# VCTR
		if tokens.peek() == 'SEP_VCTR':
			if vctr: mtrx.append(vctr.check())
			vctr = Vector()
			tokens.next()
			bindings.append(VSAME)
			continue

		# MTRX
		if tokens.peek() == 'SEP_MTRX':
			if vctr: mtrx.append(vctr.check())
			if mtrx: yield mtrx.check()
			mtrx, vctr = Matrix(), Vector()
			tokens.next()
			continue

		if tokens.peek() == 'SEP_AXIS':
			tokens.next()
			continue

		raise SyntaxError('E_TOK')

	if vctr: mtrx.append(vctr.check())
	if mtrx: yield mtrx.check()


def parse_term (token: Token, tokens: Stream, bindings: List[str]) -> Term:
	term = Term(token)
	if tokens.peek() in MOD_KINDS:
		term.opr=tokens.next()
		t = tokens.next()
		if t.kind not in DAT_KINDS: raise SyntaxError('E_EXPR_DAT')
		if t.kind in {'VSAME', 'VAR'} and t.lexeme not in bindings: raise SyntaxError('E_VAR_BIND')
		term.append(t)
	return term.check()

class Meme(Node):
	opr: Token = TOK_SEP_MTRX
	src: Memelang

	def __init__(self, src: Memelang):
		self.src = src
		super().__init__(*parse(src))
		self.check()

	def check(self) -> 'Meme':
		for mtrx_idx, mtrx in enumerate(self):
			if not isinstance(mtrx, Matrix): raise TypeError('E_TYPE_MTRX')
			for vctr_idx, vctr in enumerate(mtrx):
				if not isinstance(vctr, Vector): raise TypeError('E_TYPE_VCTR')
				for axis_idx, axis in enumerate(vctr):
					if not isinstance(axis, Axis): raise TypeError('E_TYPE_AXIS')
					# DO VAR BIND HERE
			self[mtrx_idx].pad(Axis(
					Left(TERM_ELIDE).check(),
					Right(Term(Token('VSAME',VSAME)).check()).check(),
					opr=Token('EQL',ELIDE)
				).check()
			)

		return self


# GENERATE RANDOM MEMELANG DATA
class Fuzz():
	@staticmethod
	def datum(kind:str) -> Memelang:
		if kind=='ALNUM': return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(5))
		if kind=='QUOT': return json.dumps(''.join(random.choice(' -_+,./<>[]{}\'"!@#$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(10)))
		if kind=='FLOAT': return str(random.uniform(-9, 9))
		if kind=='VAR': return SIGIL + Fuzz.datum('ALNUM')
		if kind=='EMB': return '[' + ','.join([str(random.uniform(0,1)) for _ in range(4)]) + ']'

	@staticmethod
	def lterm(mode: str) -> Memelang:
		term = ''
		if mode=='VEC': term += random.choice(['<=>','<->','<#>']) + Fuzz.datum('EMB') 
		return term

	@staticmethod
	def rterm(mode: str) -> Memelang:
		if mode == 'NUM':
			term = Fuzz.datum('FLOAT')
			if random.randint(0,1): term += random.choice(['+','-','*','/']) + Fuzz.datum('FLOAT')
		elif mode == 'STR': term = random.choice([Fuzz.datum('QUOT'), Fuzz.datum('ALNUM')])
		elif mode=='VEC': term = Fuzz.datum('EMB')
		return term

	@staticmethod
	def junc(mode: str) -> Memelang:
		if mode == 'STR': return SEP_OR.join(Fuzz.rterm(mode) for _ in range(random.randint(1,5)))
		else: return Fuzz.rterm(mode)

	@staticmethod
	def limit(bindings: List[str]|None = None) -> Memelang:
		if not bindings: bindings = []

		mode: str
		data: Memelang = ''

		comp = random.choice(['=','!=','>','<','<=','>='])

		if comp in {'<','<=','>','>='}: mode = 'NUM' if random.randint(0, 3) else 'VEC'
		else: mode='STR'

		return Fuzz.lterm(mode) + comp + Fuzz.junc(mode)

	@staticmethod
	def vector(limit_len:int = 4) -> Memelang:
		bindings, vector = [], []
		for i in range(limit_len):
			if i>0: bindings.append(VSAME)
			vector.append(Fuzz.limit(bindings))
		return SEP_AXIS.join(vector)

	@staticmethod
	def mtrx_table(col_len:int = 5) -> Memelang:
		return Fuzz.datum('ALNUM') + SEP_AXIS + VAL + SEP_AXIS + SEP_VCTR.join(Fuzz.datum('ALNUM') + SEP_AXIS + Fuzz.limit() for _ in range(col_len)) + SEP_MTRX



### SQL ### 

'''
1. EXAMPLE QUERY
MEMELANG: roles _ actor "Mark Hamill",Mark; movie _; rating >4;;
SQL: SELECT t0.actor, t0.movie, t0.rating FROM roles as t0 WHERE (t0.actor = 'Mark Hamill' or t0.actor = 'Mark') AND t0.rating > 4;

2. EXAMPLE JOIN
MEMELANG: roles _ actor "Mark Hamill"; movie _; !@ @ @; actor _;;
SQL: SELECT t0.id, t0.actor, t0.movie, t1.movie, t1.actor FROM roles AS t0, roles AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.id != t0.id AND t1.movie = t0.movie;

3. EXAMPLE TABLE JOIN WHERE ACTOR NAME = MOVIE TITLE
MEMELANG: actors _ age >21; name _; roles _ title @;;
MEMELANG(2): actors _ age >21; name:$n; roles _ title $n;;
SQL: SELECT t0.id, t0.name, t0.age, t1.title FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t1.title = t0.name;

4. EXAMPLE EMBEDDING
MEMELANG: movies _ description <=>[0.1,0.2,0.3]:dsc>0.5; year >2005; %q lmt 10; beg 100;;
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
	def holder(token: Token, bindings: dict) -> SQL:
		if token.kind == 'DBCOL': return token.datum
		elif token.kind == 'VAL': return SQLUtil.holder(bindings[VAL], bindings)
		elif token.kind == 'VSAME': return SQLUtil.holder(bindings[VSAME], bindings)
		elif token.kind == 'VAR':
			if token.lexeme not in bindings: raise SyntaxError('E_VAR_BIND')
			return SQLUtil.holder(bindings[token.lexeme], bindings)
		return '%s'

	@staticmethod
	def param(token: Token, bindings: dict) -> None|Param:
		if token.kind == 'DBCOL': return None
		elif token.kind == 'VAL': return SQLUtil.param(bindings[VAL], bindings)
		elif token.kind == 'VSAME': return SQLUtil.param(bindings[VSAME], bindings)
		elif token.kind == 'VAR':
			if token.lexeme not in bindings: raise SyntaxError('E_VAR_BIND')
			return SQLUtil.param(bindings[token.lexeme], bindings)
		return token.datum

	@staticmethod
	def term(term: Term, bindings: dict) -> Tuple[SQL, List[None|Param]]:
		sqlterm = SQLUtil.holder(term[0], bindings)
		sqlparams = [SQLUtil.param(term[0], bindings)]
		if term.opr.kind!='SEP_TOK':
			sqlterm += term.opr.lexeme + SQLUtil.holder(term[1], bindings)
			sqlparams.append(SQLUtil.param(term[1], bindings))

		return sqlterm, sqlparams

	@staticmethod
	def select(axis: Axis, bindings: dict) -> Tuple[SQL, List[None|Param], Agg]:
		agg_func = {'sum': 'SUM', 'avg': 'AVG', 'min': 'MIN', 'max': 'MAX'}
		agg = ANONE
		sqlterm, sqlparams = SQLUtil.term(axis[L][L], bindings)
		for t in axis[L][R:]:
			if t.lexeme in agg_func:
				if agg: raise SyntaxError('E_DBL_AGG')
				agg = AHAV
				sqlterm = agg_func[t.lexeme] + '(' + sqlterm + ')'
			elif t.lexeme == 'grp': agg = AGRP
			# TO DO: FLAG CONFLICTS
		return sqlterm, sqlparams, agg
		
	@staticmethod
	def where(axis: Axis, bindings: dict) -> Tuple[SQL, List[None|Param], Agg]:
		if axis.opr.kind == 'SEP_PASS': return '', [], ANONE
		sym = SQLUtil.cmp2sql[axis.opr.kind]
		lp, rp, junc, rights = '', '', '', []

		if len(axis[R]) > 1:
			lp, rp = '(', ')'
			junc = ' AND ' if axis.opr.kind == 'NOT' else ' OR '

		leftsql, params, agg = SQLUtil.select(axis, bindings)
		for right in axis[R]:
			sql, subparams = SQLUtil.term(right, bindings)
			if sym in ('LIKE','ILIKE'): sql = sql.replace('%s', "CONCAT('%', %s, '%')", 1)
			rights.append(f"{leftsql} {sym} {sql}")
			params.extend(subparams)

		return lp + junc.join(rights) + rp, params, agg

	@staticmethod
	def deref(axis: Axis, bindings: dict) -> Tuple[bool, Token]:
		if axis.singleton.kind == 'VSAME': return True, bindings.get(VSAME)
		return (bindings.get(VSAME) is not None and axis.singleton == bindings.get(VSAME)), axis.singleton


# TRANSLATE TO POSTGRES
class MemePGSQL(Meme):
	def select(self) -> List[Tuple[SQL, List[Param]]]:
		tbl_idx: int = 0
		sqls: List[Tuple[SQL, List[Param]]] = []
		axes = ('val','col','row','tbl')
		
		for mtrx in self:
			selectall = False
			tbl_alias = None
			limitstr = ''
			froms, wheres, selects, orderbys, groupbys, havings, bindings = [], [], [], [], [], [], {}
			prev = {aname:None for aname in axes}
			metas = {'val':0,'col':1,'row':2,'tbl':3,'lmt':0,'beg':0,'pri':'id'}

			# META VECTORS
			# ; %q lmt 10; beg 100
			meta_mode = False
			for vctr in mtrx:
				if len(vctr)>=AXIS_QI+1: meta_mode = (vctr[AXIS_QI].singleton.lexeme == META+'q')
				if meta_mode:
					if len(vctr)<=AXIS_QK: raise SyntaxError('E_META_LEN')
					key, val = vctr[AXIS_QK].singleton, vctr[AXIS_QV].singleton
					if key.lexeme in metas and isinstance(val.datum, type(metas[key.lexeme])): metas[key.lexeme] = val.datum
					else: raise SyntaxError('E_META')					

			# FILTER VECTORS
			# tbl row col term:meta>term,term;
			meta_mode = False
			for vctr in mtrx:
				if len(vctr)>=AXIS_QI+1: meta_mode = (vctr[AXIS_QI].singleton.lexeme == META+'q')
				if meta_mode: continue
				if len(vctr)!=4: raise SyntaxError('E_SQL_VCTR_LEN')

				curr = {aname: None for aname in axes}
				same = {aname: None for aname in axes}
				for aname in axes: same[aname], curr[aname] = SQLUtil.deref(vctr[metas[aname]], {VSAME: prev[aname]})

				valaxis = vctr[metas['val']]

				# JOIN
				if not same['tbl'] or not same['row']:

					if selectall: selects.append((f'{tbl_alias}.*', [], ANONE))
					selectall = False

					# TBL
					if not curr['tbl'] or curr['tbl'].kind != 'ALNUM': raise SyntaxError('E_TBL_ALNUM')
					tbl_alias = f't{tbl_idx}'
					froms.append(f"{curr['tbl']} AS {tbl_alias}")
					tbl_idx += 1
					pricol = f"{tbl_alias}.{metas['pri']}"

					# ROW
					bindings[VSAME]=prev['row'] if prev['row'] is not None else None
					curr['row']=bindings[VAL]=Token('DBCOL', pricol)
					where, param, _ = SQLUtil.where(vctr[metas['row']], bindings)
					if where: wheres.append((where, param, ANONE))

					selects.extend([(f"'{curr['tbl'].lexeme}' AS _a3", [], ACNST), (f"{pricol} AS _a2", [], ANONE)])

				# COL
				if not curr['col']: raise SyntaxError('E_COL')
				elif curr['col'].kind == 'VAL' and vctr[metas['col']].opr.kind == 'SEP_PASS':
					selectall=True
					continue
				elif curr['col'].kind != 'ALNUM': raise SyntaxError('E_COL_ALNUM')

				col_name = curr['col'].datum
				col_alias = f"{tbl_alias}.{col_name}"

				# VAL
				if prev['val']: bindings[VSAME]=prev['val']
				curr['val']=bindings[VAL]=Token('DBCOL', col_alias)

				select = SQLUtil.select(valaxis, bindings)
				selects.append(select)

				if any(t.lexeme=='grp' for t in valaxis[L][R:]): groupbys.append(select)
				if any(t.lexeme=='asc' for t in valaxis[L][R:]): orderbys.append((select[0]+' ASC', select[1], select[2]))
				elif any(t.lexeme=='dsc' for t in valaxis[L][R:]): orderbys.append((select[0]+' DESC', select[1], select[2]))

				where = SQLUtil.where(valaxis, bindings)
				if where[0]:
					if where[2]==AHAV: havings.append(where)
					else: wheres.append(where)
			
				for aname in axes:
					for t in vctr[metas[aname]][L][R:]:
						if t.kind == 'VAR': bindings[t.lexeme]=curr[aname]

				prev = curr.copy()

			if groupbys: 
				if selectall: raise SyntaxError('E_SLCTALL_AGRP')
				selects = [s for s in selects if s[2]>ANONE]
			elif selectall: selects.append((f'{tbl_alias}.*', [], ANONE))

			selectstr = 'SELECT ' + ', '.join([s[0] for s in selects if s[0]])
			fromstr = ' FROM ' + ', '.join(froms)
			wherestr = '' if not wheres else ' WHERE ' + ' AND '.join([s[0] for s in wheres if s[0]])
			groupbystr = '' if not groupbys else ' GROUP BY ' + ', '.join([s[0] for s in groupbys if s[0]])
			havingstr = '' if not havings else ' HAVING ' + ' AND '.join([s[0] for s in havings if s[0]])
			orderbystr = '' if not orderbys else ' ORDER BY ' + ', '.join([s[0] for s in orderbys if s[0]])
			limitstr = '' if not metas['lmt'] else (f" LIMIT {metas['lmt']}" + ('' if not metas['beg'] else f" OFFSET {metas['beg']}"))
			params = [p for s in selects+wheres+groupbys+havings+orderbys for p in s[1] if p is not None]
			sqls.append((selectstr + fromstr + wherestr + groupbystr + havingstr + orderbystr + limitstr, params))

		return sqls
