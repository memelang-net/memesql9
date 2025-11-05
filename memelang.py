'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs
MEMELANG USES AXES ORDERED HIGH TO LOW
ONE OR MORE WHITESPACES *ALWAYS* MEANS "NEW AXIS"
NEVER SPACE BETWEEN OPERATOR/COMPARATOR/COMMA/FUNC AND VALUES
'''

MEMELANG_VER = 9.44

syntax = '[table WS] [column WS] ["<=>" "\"" string "\""] [":" "$" var][":" ("min"|"max"|"cnt"|"sum"|"avg"|"last"|"grp")][":" ("asc"|"des")] [("="|"!="|">"|"<"|">="|"<="|"~"|"!~") (string|int|float|("$" var)|"@"|"_")] ";"'

examples = '''
%tab roles id :TYP=INT;>0;rating :DESC="Decimal 0-5 star rating of performance";:TYP=DEC;>0;<=5;actor :DESC="Actor's full name";:TYP=STR;movie :DESC="Movie's full name";:TYP=STR;character :DESC="Character's full name";:TYP=STR;;
%tab actors id :TYP=INT;>0;name :DESC="Actor's full name";:TYP=STR;age :DESC="Actor's age in years";:TYP=INT;>=0;<200;;
%tab movies id :TYP=INT;>0;description :DESC="Brief description of movie plot";:TYP=STR;year :DESC="Year of production AD";:TYP=INT;>1800;<2100;genre scifi,drama,comedy,documentary;:TYP=STR;title :DESC="Full movie title";:TYP=STR;;
%for actors name _;roles actor @;;
%for movies title _;roles movie @;;
%uni roles id;;
%uni roles movie;roles actor;roles character;;
%uni actors id;;
%uni movies id;;

// All movies
movies _ _;;

// all films
movies _ _;;

// all roles
roles _ _;;

// Titles and descriptions for movies
movies title _;description _;;

// actor name and ages
actors name _;age _;;

// actors age 41 years or older
actors age >=41;_;;

// Role 567 and 9766324436
roles id 567,9766324436;_;;

// Films with dystopian society narratives sim>.33
movies description <=>"dystopian">0.33;_;;

// Movies titled with Star released in 1977 or 1980
movies title ~"Star";year 1977,1980;_;;

// Actors named like Ana aged 20 to 35 inclusive
actors name ~"Ana";age >=20;<=35;_;;

// Roles rated below 1.5 for movies before 1980
movies year <1980;title _;roles movie @;rating <1.5;_;;

// roles sort rating des, movie des
roles rating :des;movie :des;;

// All movies before 1970 ordered by year asc
movies year :asc<1970;_;;

// average performer rating at least 4.2
roles rating :avg>=4.2;actor :grp;;

// minimum role rating by actor, low to high
roles rating :min:asc;actor :grp;;

// roles in movies mentioning robot rated 3+
movies description <=>"robot">=$sim;title _;roles movie @;rating >=3;;

// Costars seen with Bruce Willis and Uma Thurman
roles actor :$a~"Bruce Willis","Uma Thurman";movie _;@ @ @;actor !$a;;

// War stories before 1980: top 12 movies by minimum role rating
movies year <1980;description <=>"war">=$sim;title :grp;roles movie @;rating :min:des;%m lim 12;;

// Roles for movies Hero or House of Flying Daggers where actor name includes Li, actor A-Z
movies title "Hero","House of Flying Daggers";roles movie @;actor :asc~"Li";;
'''

import random, re, json, sys
from typing import List, Iterator, Iterable, Dict, Tuple, Union

Memelang = str
Err = SyntaxError

ELIDE, SIGIL, WILD, MSAME, SAME, MODE, EOF =  '', '$', '_', '^', '@',  '%', None
SA, SV, SM, SF, OR, PRETTY = ' ', ';', ';;', ':', ',', ' '
L, R = 0, 1

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOT',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS "John \"Jack\" Kennedy"
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
	('DSML',		r'!~'),
	('NOT',			r'!=?'),
	('EQL',			r'='),
	('GT',			r'>'),
	('LT',			r'<'),
	('SMLR',		r'~'),
	('WILD',		re.escape(WILD)),		# NEVER QUOTE
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
DELIDE = {'SAME':SAME,'MSAME':MSAME,'WILD': WILD,'EQL': '='}

Q, M = MODE+'q', MODE+'m'
VOCAB = {
	MODE+'tab': { # Table
		'CMP': {'EQL','NOT','GT','GE','LT','LE'},
		'MOD': {},
		'DAT': {'ALNUM','QUOT','INT','DEC','SAME','MSAME','WILD'},
		'FUNC': {'TYP','ROL','DESC'}
	},
	MODE+'for': { # Foreign
		'CMP': {'EQL'},
		'MOD': {},
		'DAT': {'ALNUM','QUOT','SAME','MSAME','WILD'},
		'FUNC': {}
	},
	MODE+'uni': { # Unique
		'CMP': {'EQL'},
		'MOD': {},
		'DAT': {'ALNUM','QUOT'},
		'FUNC': {}
	},
	MODE+'pri': { # Pri
		'CMP': {'EQL'},
		'MOD': {},
		'DAT': {'ALNUM','QUOT'},
		'FUNC': {}
	},
	Q: { # DQL
		'CMP': {'EQL','NOT','GT','GE','LT','LE','SMLR','DSML'},
		'MOD': {'MUL','ADD','SUB','DIV','MOD','POW','L2','IP','COS'},
		'DAT': {'ALNUM','QUOT','INT','DEC','VAR','SAME','MSAME','WILD','EMB','YMD','YMDHMS'},
		'FUNC': {"grp","asc","des","sum","avg","min","max","cnt","last"}
	},
	M: { # META
		'CMP': {'EQL'},
		'MOD': {},
		'DAT': {'ALNUM','INT'}
	}
}

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

TERM_ELIDE = Term(Token('WILD',ELIDE))


# TERM {OR TERM}
class Right(Node):
	opr: Token = TOK_OR
	def check(self) -> 'Right':
		if not len(self): return self
		if any(not isinstance(t, Term) for t in self): raise Err('E_RIGHT_TERM')
		return self


# TERM {FUNC VAR|ALNUM}
class Left(Node): 
	opr: Token = TOK_SF
	def check(self) -> 'Left':
		if not len(self): return self
		if not isinstance(self[0], Term): raise Err('E_LEFT_TERM')
		if len(self)>1 and any(t.kind not in {'ALNUM','VAR'} for t in self[1:]): raise Err('E_LEFT_FUNC')
		return self


# LEFT CMP RIGHT
class Axis(Node):
	opr: Token = TOK_EQL
	def check(self) -> 'Axis': 
		if len(self)!=2: raise Err('E_NODE_LIST')
		if not isinstance(self[0], Left): raise Err('E_AXIS_LEFT')
		if not isinstance(self[1], Right): raise Err('E_AXIS_RIGHT')
		return self
	@property
	def single(self) -> Token:
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
			elif diff<0: raise Err('E_FIRST_VECTOR_ALWAYS_LONGEST')


def lex(src: Memelang) -> Iterator[Token]:
	for m in MASTER_PATTERN.finditer(src):
		kind = m.lastgroup
		if kind in IGNORE_KINDS: continue
		if kind=='MISMATCH': raise Err(f'E_TOK {m}')
		yield Token(kind, m.group())


def parse(src: Memelang, mode: str = Q) -> Iterator[Matrix]:
	tokens = Stream(lex(src))
	bind: List[str] = []
	mat, vec = Matrix(), Vector()
	line = 1
	
	while tokens.peek():

		# AXIS
		axis = Axis(Left(), Right())

		# MODE
		if tokens.peek()=='MODE':
			if vec: raise Err('E_VEC_MODE')
			mode = tokens.next().lex
			if mode not in VOCAB: raise Err(f'E_MODE {mode}')

		# LEFT
		if tokens.peek() in VOCAB[mode]['MOD']: axis[L].append(parse_term(Token('WILD', ELIDE), tokens, bind, mode))

		# FUNC
		while tokens.peek()=='SF':
			if not axis[L]: axis[L].append(TERM_ELIDE)
			tokens.next()
			t = tokens.next()
			if t.kind=='VAR': bind.append(t.lex)
			elif t.lex not in VOCAB[mode]['FUNC']: raise Err(f'E_FUNC_NAME :{t.lex}')
			axis[L].append(t)
			
		# CMP
		if tokens.peek() in VOCAB[mode]['CMP']:
			axis.opr=tokens.next()
			if tokens.peek() not in VOCAB[mode]['DAT']: raise Err(f'E_CMP_DAT {tokens.next()}')

		# RIGHT
		while tokens.peek() in VOCAB[mode]['DAT']:
			if not axis[L]: axis[L].append(TERM_ELIDE)
			axis[R].append(parse_term(tokens.next(), tokens, bind, mode))
			if tokens.peek()=='OR':
				tokens.next()
				if tokens.peek() not in VOCAB[mode]['DAT']: raise Err('E_OR_TRAIL')
			if tokens.peek() == 'MODE': raise Err('E_RIGHT_MODE')

		if axis[L] and not axis[R]: axis[R]=Right(Term(Token('WILD',ELIDE)))

		if axis[R]:
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
			mode = Q
			continue

		if tokens.peek()=='SA':
			tokens.next()
			continue

		raise Err(f'E_PARSE {tokens.next()}\n{vec}')

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
		inp: str = tok.lex
		out: str = json.dumps([0.1,0.2])
		return Token('EMB', out)



### SQL ### 

ANONE, ACNST, AGRP, AHAV = 0, 1, 2, 3
PH, EPH = '%s', ' = %s'

class SQL():
	cmp2sql = {'EQL':'=','NOT':'!=','GT':'>','GE':'>=','LT':'<','LE':'<=','SMLR':'ILIKE','DSML':'NOT ILIKE'}
	lex: str
	alias: str
	params: List[Union[int, float, str, list]]
	agg: int|None
	tally = {'cnt':'COUNT(1)','sum': 'SUM', 'avg': 'AVG', 'min': 'MIN', 'max': 'MAX', 'last': 'MAX'}

	def __init__(self, lex: str = '', params: List[Union[int, float, str, list]] = None, agg: int = ANONE, alias: str = ''):
		self.lex = lex
		self.agg = agg
		self.alias = alias
		self.params = [] if params is None else params

	@staticmethod
	def dat(token: Token, bind: dict) -> 'SQL':
		if token.kind in {'WILD','SAME','MSAME','VAR'}:
			if token.delide not in bind: raise Err('E_VAR_BIND')
			return SQL() if bind[token.delide] is None else bind[token.delide]
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
		left = SQL.term(axis[L][L], bind)
		for t in axis[L][R:]:
			if t.lex in SQL.tally:
				if left.agg == AHAV: raise Err('E_DBL_AGG')
				left.agg = AHAV
				left.lex = SQL.tally[t.lex] + ('' if '(1)' in SQL.tally[t.lex] else '(' + left.lex + ')')
			elif t.lex=='grp': left.agg = AGRP
			# TO DO: FLAG CONFLICTS
		if alias: left.alias=alias
		return left
		
	@staticmethod
	def where(axis: Axis, bind: dict) -> 'SQL':
		if axis.single.kind==WILD: return SQL()
		sym = SQL.cmp2sql[axis.opr.kind]
		lp, rp, right, ts = '', '', '', []
		params = []

		if len(axis[R]) > 1:
			lp, rp = '(', ')'
			right = ' AND ' if axis.opr.kind in ('NOT','DSML') else ' OR '

		select = SQL.select(axis, bind)
		params.extend(select.params)
		for t in axis[R]:
			where = SQL.term(t, bind)
			if 'ILIKE' in sym: where.lex = where.lex.replace(PH, "CONCAT('%', %s, '%')")
			ts.append(f"{select.lex} {sym} {where.lex}")
			params.extend(where.params)

		return SQL(lp + right.join(ts) + rp, params, select.agg)

	@property
	def holder(self) -> str:
		return self.lex + ( '' if not self.alias else f' AS {self.alias}')

	@property
	def param(self):
		return None if self.lex!=PH or len(self.params)!=1 else self.params[0]

	# NOT DB SAFE - FOR DEBUGGING ONLY
	def __str__(self) -> str:
		sql = self.lex
		if self.alias: sql += f' AS {self.alias}'
		for p in self.params:
			if isinstance(p, str): v = "'" + p.replace("'", "''") + "'"
			elif isinstance(p, list): v = json.dumps(p, separators=(',',':')) + '::VECTOR'
			elif p is None: v = 'NULL'
			else: v = str(p)
			sql = sql.replace(PH, v, 1)
		return sql

	def __eq__(self, other): return isinstance(other, SQL) and str(self)==str(other)

# TRANSLATE TO SQL FOR POSTGRES
class MemePGSQL(Meme):
	def select(self) -> List[SQL]:
		self.embed()
		tab_idx: int = 0
		sql: List[SQL] = []
		VAL, COL, TAB = 0, 1, 2
		axes = (VAL, COL, TAB)

		for mat in self:
			sel_all, tab_alias = False, None
			froms, wheres, selects, ords, groups, havings, bind = [], [], [], [], [], [], {}
			prev = {axis:None for axis in axes}
			config = {M: {'lim':0,'beg':0,'sim':0.5}}
			for k in config[M]: bind[SIGIL+k]=SQL(PH, [config[M][k]])

			# Precheck if grouped
			grouping = False
			for vec in mat:
				if vec.mode!=Q or len(vec)<3: continue
				if 'grp' in set(t.lex for t in vec[VAL][L][R:]):
					grouping = True
					break

			for vec in mat:

				# META VECTORS
				# ; %m lim 10; beg 100
				if vec.mode==M:
					if len(vec)!=2: raise Err('E_X_LEN')
					key, val = vec[1].single, vec[0].single
					if key.lex in config[M] and isinstance(val.dat, type(config[M][key.lex])):
						config[M][key.lex] = val.dat
						bind[SIGIL+key.lex] = SQL(PH, [val.dat])
					else: raise Err('E_MODE_KEY')
					continue

				# QUERY VECTORS
				# tab col term:func:func>term,term;
				if vec.mode!=Q: continue
				if len(vec)<3: raise Err('E_Q_LEN')

				curr = {name: None for name in axes}
				
				# TAB
				bind[WILD], bind[SAME] = None, prev[TAB]
				curr[TAB] = SQL.single(vec[TAB], bind)

				# JOIN
				if vec[TAB].single.lex != ELIDE:
					if curr[TAB].param is None: raise Err('E_TBL_NAME')
					tab_alias = f't{tab_idx}'
					tab_idx += 1
					froms.append(SQL(curr[TAB].param, [], ACNST, tab_alias))

				# COL
				if vec[COL].single.lex in (WILD,ELIDE) and vec[VAL].single.lex==WILD:
					sel_all=True
					continue

				bind[SAME], bind[WILD] = prev[COL], None
				curr[COL] = SQL.single(vec[COL], bind)
				if curr[COL].param is None: raise Err('E_COL_NAME')
				#selects.append(curr[COL])

				# VAL
				bind[SAME],bind[WILD] = prev[VAL], SQL(tab_alias + '.' + curr[COL].params[0])
				curr[VAL] = SQL.single(vec[VAL], bind)

				if vec[VAL].single.kind!='WILD': wheres.append(SQL.where(vec[VAL], bind))

				funcs = set(t.lex for t in vec[VAL][L][R:])
				if grouping and not bool(funcs & (set(SQL.tally) | {'grp'})): 
					vec[VAL][L].append(Token('ALNUM', 'last'))
					if vec[VAL].opr.lex == ELIDE: vec[VAL].opr = Token('EQL','=')

				select = SQL.select(vec[VAL], bind) # f"{tab_alias}_{curr[COL].param}"
				selects.append(select)

				# AGG/SORT
				if 'grp' in funcs: groups.append(select)
				if 'asc' in funcs: ords.append(SQL(select.lex+' ASC', select.params, ANONE))
				elif 'des' in funcs: ords.append(SQL(select.lex+' DESC', select.params, ANONE))

				# BIND VARS			
				for axis in axes:
					for t in vec[axis][L][R:]:
						if t.kind=='VAR': bind[t.lex]=curr[axis]

				prev = curr.copy()

			if grouping: 
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
				itemstr = sep.join([(s.holder if usealias else s.lex)  for s in items if s.lex])
				if not itemstr: continue
				sqlstr+=space+keyword+' '+itemstr
				params.extend(p for s in items for p in s.params if p is not None)
				space = ' '

			if config[M]['lim']: sqlstr += f" LIMIT {config[M]['lim']}"
			if config[M]['beg']: sqlstr += f" OFFSET {config[M]['beg']}"

			sql.append(SQL(sqlstr, params))

		return sql


### CLI ###

if __name__ == "__main__":
	lines = []
	if len(sys.argv)==2:
		meme = MemePGSQL(sys.argv[1])
		print(str(meme))
		print(meme.select())
	elif len(sys.argv)==3 and sys.argv[1]=='file':
		with open(sys.argv[2], encoding='utf-8') as f: lines = [l.rstrip() for l in f]
	else: lines = examples.splitlines()

	if lines:
		for i in range(len(lines) - 1):
			if lines[i].startswith('// '):
				print(f'//{i}{lines[i]}')
				meme = MemePGSQL(lines[i + 1])
				print(str(meme))
				print()
