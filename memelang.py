'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs
AXIS:0 IS ORDERED HIGH TO LOW
ONE OR MORE WHITESPACES *ALWAYS* MEANS "NEW FIELD"
NEVER SPACE BETWEEN OPERATOR/COMPARATOR/COMMA/FUNC AND VALUES
'''

MEMELANG_VER = 9.48

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

// Every film
movies _ _;;

// Roles
roles _ _;;

// Titles and descriptions for movies
movies title _;description _;;

// Actor name and ages
actors name _;age _;;

// Actors age 41 years or older
actors age >=41;_;;

// Role 567 and 9766324436
roles id 567,9766324436;_;;

// Films with dystopian society narratives sim>.33
movies description <=>"dystopian"<0.33;_;;

// Movies titled with Star released in 1977 or 1980
movies title ~"Star";year 1977,1980;_;;

// Actors named like Ana aged 20 to 35 inclusive
actors name ~"Ana";age >=20;<=35;_;;

// Roles rated below 1.5 for movies before 1980
movies year <1980;title _;roles movie @;rating <1.5;_;;

// Roles sort rating des, movie descending
roles rating :des;movie :des;;

// All movies before 1970 ordered by year asc
movies year :asc<1970;_;;

// Average performer rating at least 4.2
roles rating :avg>=4.2;actor :grp;;

// Minimum role rating by actor, low to high
roles rating :min:asc;actor :grp;;

// Roles in movies mentioning robot rated 3+
movies description <=>"robot"<=$sim;title _;roles movie @;rating >=3;;

// Costars seen with Bruce Willis and Uma Thurman
roles actor :$a~"Bruce Willis","Uma Thurman";movie _;@ @ @;actor !$a;;

// War stories before 1980: top 12 movies by minimum role rating
movies year <1980;description <=>"war"<=$sim;title :grp;roles movie @;rating :min:des;%m lim 12;;

// Roles for movies Hero or House of Flying Daggers where actor name includes Li, actor A-Z
movies title "Hero","House of Flying Daggers";roles movie @;actor :asc~"Li";;
'''

import random, re, json, sys
from typing import List, Iterator, Iterable, Dict, Tuple, Union

Memelang = str
Err = SyntaxError

ELIDE, SIGIL, WILD, MSAME, SAME, MODE, EOF =  '', '$', '_', '^', '@',  '%', None
SAX0, SAX1, SAX2, FLAG, OR, PRETTY = ' ', ';', ';;', ':', ',', ' '
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
	('FLAG',		re.escape(FLAG) + r'[a-z]+'),
	('BIND',		re.escape(FLAG+SIGIL) + r'[A-Za-z0-9_]+'),
	('VAR',			re.escape(SIGIL) + r'[A-Za-z0-9_]+'),
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
	('YMDHMS',		r'\d\d\d\d\-\d\d-\d\d\-\d\d:\d\d:\d\d'),	 	# YYYY-MM-DD-HH:MM:SS
	('YMD',			r'\d\d\d\d\-\d\d\-\d\d'),	 					# YYYY-MM-DD
	('ALNUM',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERICS ARE UNQUOTED
	('DEC',			r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('SUB',			r'\-'), # AFTER INT/DEC
	('SAX2',		re.escape(SAX2)),
	('SAX1',		re.escape(SAX1)),
	('OR',			re.escape(OR)),
	('SAX0',		r'\s+'),
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
		'FLAG': {'TYP','ROL','DESC'}
	},
	MODE+'for': { # Foreign
		'CMP': {'EQL'},
		'MOD': {},
		'DAT': {'ALNUM','QUOT','SAME','MSAME','WILD'},
		'FLAG': {}
	},
	MODE+'uni': { # Unique
		'CMP': {'EQL'},
		'MOD': {},
		'DAT': {'ALNUM','QUOT'},
		'FLAG': {}
	},
	MODE+'pri': { # Pri
		'CMP': {'EQL'},
		'MOD': {},
		'DAT': {'ALNUM','QUOT'},
		'FLAG': {}
	},
	Q: { # DQL
		'CMP': {'EQL','NOT','GT','GE','LT','LE','SMLR','DSML'},
		'MOD': {'MUL','ADD','SUB','DIV','MOD','POW','L2','IP','COS'},
		'DAT': {'ALNUM','QUOT','INT','DEC','VAR','SAME','MSAME','WILD','EMB','YMD','YMDHMS'},
		'FLAG': {"grp","asc","des","sum","avg","min","max","cnt","last"}
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
		self.lex = lex
		self.delide = DELIDE[kind] if lex == '' and kind in DELIDE else lex
		if kind=='QUOT': 	self.dat = json.loads(lex)
		elif kind=='EMB': 	self.dat = json.loads(lex)
		elif kind=='DEC': 	self.dat = float(lex)
		elif kind=='INT':	self.dat = int(lex)
		elif kind=='NULL':	self.dat = None
		else: 				self.dat = lex

	def __str__(self) -> Memelang: return self.lex
	def __eq__(self, other): return isinstance(other, Token) and self.kind==other.kind and self.lex==other.lex


TOK_NULL = Token('NULL', '')
TOK_EQL  = Token('EQL', ELIDE)
TOK_SAX0 = Token('SAX0', SAX0)
TOK_SAX1 = Token('SAX1', SAX1+PRETTY)
TOK_SAX2 = Token('SAX2', SAX2+PRETTY)


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


class TList(list):
	opr: Token = TOK_NULL
	def __init__(self, *items): super().__init__(items)
	def __str__(self) -> Memelang:
		return self.opr.lex.join(map(str, self))


# One field constrains a Table, Column, or Value
class Field():
	left: TList
	flag: TList
	comp: Token
	right: TList
	
	def __init__(self):
		self.left = TList()
		self.flag = TList()
		self.comp = TOK_EQL
		self.right = TList()

	@property
	def single(self) -> Token:
		return TOK_NULL if self.comp.kind != 'EQL' or len(self.right)!=1 else self.right[0]

	def __str__(self) -> Memelang:
		return str(self.left) + ''.join(map(str, self.flag)) + self.comp.lex + str(self.right)

class Axis(list):
	opr: Token = TOK_NULL

	def __init__(self, *items): super().__init__(items)
	def prepend(self, item): self.insert(0, item)
	def check(self) -> 'Axis': 
		if len(self)==0: raise Err('E_AXIS_EMPTY')
		return self
	@property
	def iter(self): return iter(self)
	@property
	def prefix(self) -> Memelang: return ''
	@property
	def suffix(self) -> Memelang: return ''
	
	def __str__(self) -> Memelang:
		return re.sub(r'\s+', ' ', self.prefix + self.opr.lex.join(map(str, self.iter)) + self.suffix)

	def pad(self, padding:Field) -> None:
		pass


# Axis:0 builds predicates as Axis0[2,1,0]=(Table,Column,Value)
class Axis0(Axis):
	opr: Token = TOK_SAX0
	mode: str = Q
	@property
	def iter(self): return reversed(self)
	@property
	def prefix(self) -> Memelang: return '' if self.mode == Q else (self.mode + SAX0)


# Axis:1 AND-joins predicates
class Axis1(Axis):
	opr: Token = TOK_SAX1
	def pad(self, padding:Field) -> None:
		max_len = 0
		for idx, axis0 in enumerate(self):
			if axis0.mode != Q: continue
			if not max_len: max_len = len(axis0)
			diff = max_len - len(axis0)
			if diff>0: self[idx] += [padding] * diff
			elif diff<0: raise Err('E_FIRST_VECTOR_ALWAYS_LONGEST')


def lex(src: Memelang) -> Iterator[Token]:
	for m in MASTER_PATTERN.finditer(src):
		kind = m.lastgroup
		if kind in IGNORE_KINDS: continue
		if kind=='MISMATCH': raise Err(f'E_TOK {m}')
		yield Token(kind, m.group())


def parse(src: Memelang, mode: str = Q) -> Iterator[Axis1]:
	tokens = Stream(lex(src))
	bind: List[str] = ['$sim']
	axis1, axis0 = Axis1(), Axis0()
	line = 1
	
	while tokens.peek():

		# VECTOR MODE
		if tokens.peek()=='MODE':
			if axis0: raise Err('E_VEC_MODE')
			mode = tokens.next().lex
			if mode not in VOCAB: raise Err(f'E_MODE {mode}')

		# FIELD
		field = Field()

		# LEFT
		if tokens.peek() in VOCAB[mode]['MOD']:
			field.left.append(TOK_NULL)
			field.left.opr=tokens.next()
			t = tokens.next()
			if t.kind not in VOCAB[mode]['DAT']: raise Err('E_TERM_DAT')
			if t.kind in {'SAME', 'VAR'} and t.lex not in bind: raise Err('E_VAR_BIND')
			field.left.append(t)

		# FLAG
		while tokens.peek() in {'FLAG','BIND'}:
			t = tokens.next()
			if t.kind=='BIND': bind.append(t.lex[1:])
			elif t.lex[1:] not in VOCAB[mode]['FLAG']: raise Err(f'E_FLAG_NAME {t.lex}')
			field.flag.append(t)
			
		# CMP
		if tokens.peek() in VOCAB[mode]['CMP']:
			field.comp=tokens.next()
			if tokens.peek() not in VOCAB[mode]['DAT']: raise Err(f'E_CMP_DAT {tokens.next()}')

		# RIGHT
		while tokens.peek() in VOCAB[mode]['DAT']:
			t = tokens.next()
			if t.kind in {'SAME', 'VAR'} and t.lex not in bind: raise Err('E_VAR_BIND')
			field.right.append(t)
			if tokens.peek()=='OR':
				field.right.opr=tokens.next()
				if tokens.peek() not in VOCAB[mode]['DAT']: raise Err('E_OR_TRAIL')
			if tokens.peek() == 'MODE': raise Err('E_RIGHT_MODE')

		if field.left or field.right or field.flag:
			axis0.prepend(field) # AXIS:0 HIGH->LOW
			continue

		# AXIS:1
		if tokens.peek()=='SAX1':
			if axis0: 
				axis0.mode=mode
				axis1.append(axis0.check())
			axis0 = Axis0()
			tokens.next()
			bind.append(SAME)
			continue

		# AXIS:2
		if tokens.peek()=='SAX2':
			if axis0: 
				axis0.mode=mode
				axis1.append(axis0.check())
			if axis1: yield axis1.check()
			axis1, axis0 = Axis1(), Axis0()
			tokens.next()
			mode = Q
			continue

		if tokens.peek()=='SAX0':
			tokens.next()
			continue

		raise Err(f'E_PARSE {tokens.next()}\n{axis0}')

	if axis0: 
		axis0.mode=mode
		axis1.append(axis0.check())
	if axis1: yield axis1.check()


# Axis:2 UNIONS distinct queries
class Axis2(Axis):
	opr: Token = TOK_SAX2
	src: Memelang

	def __init__(self, src: Memelang):
		self.src = src
		super().__init__(*parse(src))
		self.check()

	def check(self) -> 'Axis2':
		for axis1 in self:
			if not isinstance(axis1, Axis1): raise TypeError('E_TYPE_MAT')
			for axis0 in axis1:
				if not isinstance(axis0, Axis0): raise TypeError('E_TYPE_VEC')
				for field in axis0:
					if not isinstance(field, Field): raise TypeError('E_TYPE_LIMIT')
			pad_field=Field()
			pad_field.right=TList(Token('SAME',ELIDE))
			axis1.pad(pad_field)

		return self

	@property
	def suffix(self) -> Memelang: return SAX2

	def embed(self):
		for axis1 in self:
			for axis0 in axis1:
				for field in axis0:
					if field.left.opr.kind in {'COS','L2','IP'} and len(field.left)==2 and field.left[1].kind in {'QUOT','ALNUM'}: field.left[1] = self.embedify(field.left[1])

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
	def term(term: TList, bind: dict) -> 'SQL':
		if not term: return SQL()
		dats = [SQL.dat(t, bind) for t in term]
		return SQL(term.opr.lex.join(dat.lex for dat in dats), [p for dat in dats for p in dat.params])

	@staticmethod
	def single(field: Field, bind: dict) -> 'SQL':
		return SQL() if field.single.dat is None else SQL.dat(field.single, bind)

	@staticmethod
	def select(field: Field, bind: dict, alias: str = '') -> 'SQL':
		left = SQL.term(field.left, bind)
		for t in field.flag:
			if t.lex in SQL.tally:
				if left.agg == AHAV: raise Err('E_DBL_AGG')
				left.agg = AHAV
				left.lex = SQL.tally[t.lex] + ('' if '(1)' in SQL.tally[t.lex] else '(' + left.lex + ')')
			elif t.lex=='grp': left.agg = AGRP
			# TO DO: FLAG CONFLICTS
		if alias: left.alias=alias
		return left
		
	@staticmethod
	def where(field: Field, bind: dict) -> 'SQL':
		if field.single.kind=='WILD': return SQL()
		sym = SQL.cmp2sql[field.comp.kind]
		lp, rp, right, ts = '', '', '', []
		params = []

		if len(field.right) > 1:
			lp, rp = '(', ')'
			right = ' AND ' if field.comp.kind in ('NOT','DSML') else ' OR '

		select = SQL.select(field, bind)
		params.extend(select.params)
		where = SQL.term(field.right, bind)
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
class Axis2PGSQL(Axis2):
	def select(self) -> List[SQL]:
		self.embed()
		tab_idx: int = 0
		sql: List[SQL] = []
		VAL, COL, TAB = 0, 1, 2
		slots = (VAL, COL, TAB)

		for axis1 in self:
			sel_all, tab_alias = False, None
			froms, wheres, selects, ords, groups, havings, bind = [], [], [], [], [], [], {}
			prev = {slot:None for slot in slots}
			config = {M: {'lim':0,'beg':0,'sim':0.5}}
			for k in config[M]: bind[SIGIL+k]=SQL(PH, [config[M][k]])

			# Precheck if grouped
			grouping = False
			for axis0 in axis1:
				if axis0.mode!=Q or len(axis0)<3: continue
				if 'grp' in set(t.lex for t in axis0[VAL].flag):
					grouping = True
					break

			for axis0 in axis1:

				# META VECTORS
				# ; %m lim 10; beg 100
				if axis0.mode==M:
					if len(axis0)!=2: raise Err('E_X_LEN')
					key, val = axis0[1].single, axis0[0].single
					if key.lex in config[M] and isinstance(val.dat, type(config[M][key.lex])):
						config[M][key.lex] = val.dat
						bind[SIGIL+key.lex] = SQL(PH, [val.dat])
					else: raise Err('E_MODE_KEY')
					continue

				# QUERY VECTORS
				# tab col term:flag:flag>term,term;
				if axis0.mode!=Q: continue
				if len(axis0)<3: raise Err('E_Q_LEN')

				curr = {slot: None for slot in slots}
				
				# TAB
				bind[WILD], bind[SAME] = None, prev[TAB]
				curr[TAB] = SQL.single(axis0[TAB], bind)

				# JOIN
				if axis0[TAB].single.lex != ELIDE:
					if curr[TAB].param is None: raise Err('E_TBL_NAME')
					tab_alias = f't{tab_idx}'
					tab_idx += 1
					froms.append(SQL(curr[TAB].param, [], ACNST, tab_alias))

				# COL
				if axis0[COL].single.lex in (WILD,ELIDE) and axis0[VAL].single.lex==WILD:
					sel_all=True
					continue

				bind[SAME], bind[WILD] = prev[COL], None
				curr[COL] = SQL.single(axis0[COL], bind)
				if curr[COL].param is None: raise Err('E_COL_NAME')
				#selects.append(curr[COL])

				# VAL
				bind[SAME],bind[WILD] = prev[VAL], SQL(tab_alias + '.' + curr[COL].params[0])
				curr[VAL] = SQL.single(axis0[VAL], bind)

				if axis0[VAL].single.kind!='WILD': wheres.append(SQL.where(axis0[VAL], bind))

				flags = set(t.lex for t in axis0[VAL].flag)
				if grouping and not bool(flags & (set(SQL.tally) | {'grp'})): 
					axis0[VAL].flag.append(Token('ALNUM', 'last'))
					if axis0[VAL].comp.lex == ELIDE: axis0[VAL].comp = Token('EQL','=')

				select = SQL.select(axis0[VAL], bind) # f"{tab_alias}_{curr[COL].param}"
				selects.append(select)

				# AGG/SORT
				if 'grp' in flags: groups.append(select)
				if 'asc' in flags: ords.append(SQL(select.lex+' ASC', select.params, ANONE))
				elif 'des' in flags: ords.append(SQL(select.lex+' DESC', select.params, ANONE))

				# BIND VARS			
				for slot in slots:
					for t in axis0[slot].flag:
						if t.kind=='BIND': bind[t.lex[1:]]=curr[slot]

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
		axis2 = Axis2PGSQL(sys.argv[1])
		print(str(axis2))
		print(axis2.select())
	elif len(sys.argv)==3 and sys.argv[1]=='file':
		with open(sys.argv[2], encoding='utf-8') as f: lines = [l.rstrip() for l in f]
	else: lines = examples.splitlines()

	if lines:
		for i in range(len(lines) - 1):
			if lines[i].startswith('// '):
				print(f'//{i}{lines[i]}')
				axis2 = Axis2PGSQL(lines[i + 1])
				print(str(axis2))
				print(str(axis2.select()[0]))
				print()
