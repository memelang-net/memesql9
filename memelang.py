'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs
MEMELANG USES AXES ORDERED HIGH TO LOW
ALWAYS WHITESPACE MEANS "NEW AXIS"
NEVER SPACE AROUND OPERATOR
NEVER SPACE BETWEEN COMPARATOR/COMMA/FUNC AND VALUES
'''

MEMELANG_VER = 9.25

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
	('EMB',			r'\[(?:-?\d+(?:\.\d+)?)(?:,-?\d+(?:\.\d+)?)*\]'), # JSON ARRAY OF DECS [0.1,0.2]
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
		'DAT': {'ALNUM','QUOT','INT','DEC','VAR','SAME','MSAME','VAL','EMB'},
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
LEFT ::= TERM {SF VAR|ALNUM}
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
	def iter(self): return iter(self)
	def prefix(self) -> Memelang: return ''
	def suffix(self) -> Memelang: return ''
	def __str__(self) -> Memelang: return self.prefix() + self.opr.lex.join([s for s in map(str, self.iter()) if s]) + self.suffix()

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
	def iter(self): return reversed(self)
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
		if kind=='TOK': kind = m.group()[1:-1]
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
				if tokens.peek() == MODE: raise Err('E_RIGHT_MODE')

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
		for mat_idx, mat in enumerate(self):
			if not isinstance(mat, Matrix): raise TypeError('E_TYPE_MAT')
			for vec_idx, vec in enumerate(mat):
				if not isinstance(vec, Vector): raise TypeError('E_TYPE_VEC')
				for axis_idx, axis in enumerate(vec):
					if not isinstance(axis, Axis): raise TypeError('E_TYPE_AXIS')
					# DO VAR BIND HERE
			axis=Axis(Left(TERM_ELIDE), Right(Term(Token('SAME',ELIDE))))
			axis.opr=Token('EQL',ELIDE)
			self[mat_idx].pad(axis)

		return self

	def suffix(self) -> Memelang: return SM


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
	def lterm(mode: str) -> Memelang:
		term = ''
		if mode=='VEC': term += random.choice(['<=>','<->','<#>']) + Fuzz.dat('EMB') 
		return term

	@staticmethod
	def rterm(mode: str) -> Memelang:
		if mode=='NUM':
			term = Fuzz.dat('DEC')
			if random.randint(0,1): term += random.choice(['+','-','*','/']) + Fuzz.dat('DEC')
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
MEMELANG: actors _ age >21; name _; roles _ title @;;
MEMELANG(2): actors _ age >21; name:$n; roles _ title $n;;
SQL: SELECT t0.id, t0.name, t0.age, t1.title FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t1.title = t0.name;

5. EXAMPLE EMBEDDING
MEMELANG: movies _ description <=>[0.1,0.2,0.3]:dsc>0.5; year >2005; %m lim 10; beg 100;;
SQL: SELECT t0.id, t0.description<=>[0.1,0.2,0.3], t0.year from movies AS t0 WHERE t0.description<=>[0.1,0.2,0.3]>0.5 AND t0.year>2005 ORDER BY t0.description<=>[0.1,0.2,0.3] DESC LIMIT 10 OFFSET 100;

6. EXAMPLE AGGREGATION
MEMELANG: roles _ rating :avg; actor :grp="Mark Hamill","Carrie Fisher";;
SQL: SELECT AVG(t0.rating), t0.actor FROM roles AS t0 WHERE (t0.actor = 'Mark Hamill' OR t0.actor = 'Carrie Fisher') GROUP BY t0.actor;
'''

SQL = str
Param = int|float|str|list
Agg = int
ANONE, ACNST, AGRP, AHAV = 0, 1, 2, 3

class SQLUtil():
	cmp2sql = {'EQL':'=','NOT':'!=','GT':'>','GE':'>=','LT':'<','LE':'<=','SMLR':'ILIKE'}

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
		agg_func = {'cnt':'COUNT(1)','sum': 'SUM', 'avg': 'AVG', 'min': 'MIN', 'max': 'MAX'}
		agg = ANONE
		sqlterm, sqlparams = SQLUtil.term(axis[L][L], bind)
		for t in axis[L][R:]:
			if t.lex in agg_func:
				if agg: raise Err('E_DBL_AGG')
				agg = AHAV
				sqlterm = agg_func[t.lex] + ('' if '(1)' in agg_func[t.lex] else '(' + sqlterm + ')')
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
			config = {M: {'val':0,'col':1,'row':2,'tab':3,'lim':0,'beg':0,'pri':'id'}}

			# CONFIG VECTORS
			# ; %m lim 10; beg 100
			for vec in mat:
				if vec.mode!=M: continue
				if len(vec)!=2: raise Err('E_X_LEN')
				key, val = vec[1].single, vec[0].single
				if key.lex in config[M] and isinstance(val.dat, type(config[M][key.lex])): config[M][key.lex] = val.dat
				else: raise Err('E_MODE_KEY')					

			# QUERY VECTORS
			# tab row col term:meta>term,term;
			for vec in mat:
				if vec.mode!=Q: continue
				if len(vec)<4: raise Err('E_Q_LEN')

				bind[VAL]=None
				curr = {name: None for name in axes}
				for name in axes:
					axis = vec[config[M][name]]
					bind[SAME] = prev[name]
					curr[name] = bind[axis.single.delide] if axis.single.delide in bind else axis.single

				bind[SAME] = None
				colaxis = vec[config[M]['col']]
				valaxis = vec[config[M]['val']]

				# JOIN
				if prev['tab']!=curr['tab'] or prev['row']!=curr['row']:

					if selectall: selects.append((f'{tab_alias}.*', [], ANONE))
					selectall = False

					# TAB
					if not curr['tab'] or curr['tab'].kind!='ALNUM': raise Err('E_TBL_ALNUM')
					tab_alias = f't{tab_idx}'
					froms.append(f"{curr['tab']} AS {tab_alias}")
					tab_idx += 1
					pricol = f"{tab_alias}.{config[M]['pri']}"

					# ROW
					bind[SAME]=prev['row'] if prev['row'] is not None else None
					curr['row']=bind[VAL]=Token('DBCOL', pricol)
					where, param, _ = SQLUtil.where(vec[config[M]['row']], bind)
					if where: wheres.append((where, param, ANONE))

					selects.extend([(f"'{curr['tab'].lex}' AS _tab", [], ACNST), (f"{pricol} AS _row", [], ANONE)])

				# COL
				if colaxis.single.kind=='VAL':
					selectall=True
					continue
				elif not curr['col']: raise Err('E_COL_NONE')
				elif curr['col'].kind=='ALNUM': col_alias = tab_alias + '.' + curr['col'].dat
				else: raise Err('E_COL_ALNUM')

				# VAL
				if prev['val']: bind[SAME]=prev['val']
				curr['val']=bind[VAL]=Token('DBCOL', col_alias)

				select = SQLUtil.select(valaxis, bind)
				selects.append(select)

				# AGG/SORT
				funcs = set(t.lex for t in valaxis[L][R:])
				if 'grp' in funcs: groups.append(select)
				if 'asc' in funcs: ords.append((select[0]+' ASC', select[1], select[2]))
				elif 'dsc' in funcs: ords.append((select[0]+' DESC', select[1], select[2]))

				# HAVING
				where = SQLUtil.where(valaxis, bind)
				if where[0]:
					if where[2]==AHAV: havings.append(where)
					else: wheres.append(where)
			
				for name in axes:
					for t in vec[config[M][name]][L][R:]:
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
			if config[M]['lim']: limstr += f" LIMIT {config[M]['lim']}"
			if config[M]['beg']: limstr += f" OFFSET {config[M]['beg']}"
			params = [p for s in selects+wheres+groups+havings+ords for p in s[1] if p is not None]
			sql.append((selectstr + fromstr + wherestr + groupstr + havingstr + ordstr + limstr, params))

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
			print(f"{idx} {example['input']}")
			print(example['output'])
			meme = MemePGSQL(example['output'])
			print(str(meme))
			print(meme.select())
			print()
		print('SUCCESS')
	else: raise Err('E_ARG')
