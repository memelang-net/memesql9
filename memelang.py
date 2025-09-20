'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs

MEMELANG USES AXES
AXES ORDERED HIGH TO LOW
NEVER SPACES BETWEEN COMPARATOR/COMMA AND VALUES
SPACE MEANS "NEW AXIS"
'''

MEMELANG_VER = 9.08

import random, re, json, operator
from typing import List, Iterator, Iterable, Dict, Tuple, Any, Union

Axis, Memelang = int, str

ELIDE = ''
SIGIL, VAL, MSAME, VSAME, ASSN, EOF =  '$', '_', '^', '@', ':', None
SEP_LIMIT, SEP_VCTR, SEP_MTRX, SEP_OR = ' ', ';', ';;', ','
SEP_VCTR_PRETTY, SEP_MTRX_PRETTY = ' ; ', ' ;;\n'
LEFT, RIGHT, AVAR = 0, 1, 2

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOT',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS "John \"Jack\" Kennedy"
	('MTBL',		r'-*\|'),
	('EMB',			r'\[(?:-?\d+(?:\.\d+)?)(?:,-?\d+(?:\.\d+)?)*\]'), # JSON ARRAY OF FLOATS [0.1,0.2]
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
	#('META',		r'`'),
	('ASSN',		re.escape(ASSN)),
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
MOD_KINDS = {'MUL':{'NUM'},'ADD':{'NUM'},'SUB':{'NUM'},'DIV':{'NUM'},'MOD':{'NUM'},'POW':{'NUM'},'L2':{'EMB'},'IP':{'EMB'},'COS':{'EMB'},'TSQ':{'TSQ'}}
DATUM_KINDS = {'ALNUM','QUOT','INT','FLOAT','VAR','VSAME','MSAME','VAL','EMB'}
IGNORE_KINDS = {'COMMENT','MTBL'}

EBNF = '''
TERM ::= DATUM [MOD DATUM]
JUNC ::= {TERM} {SEP_OR {TERM}}
LIMIT ::= [TERM] [CMP] [JUNC]
VCTR ::= LIMIT {SEP_LIMIT LIMIT}
MTRX ::= VCTR {SEP_VCTR VCTR}
MEME ::= MTRX {SEP_MTRX MTRX}
'''

class Token():
	kind: str
	kinds: list[str]
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
		else: 					self.datum = lexeme

	def dump(self) -> Union[str, float, int, list]: return self.datum
	def __str__(self) -> Memelang: return self.lexeme
	def __eq__(self, other): return isinstance(other, Token) and self.kind == other.kind and self.lexeme == other.lexeme


TOK_EQL = Token('EQL', ELIDE)
TOK_NOT = Token('NOT', '!')
TOK_GT = Token('GT', '>')
TOK_SEP_LIMIT = Token('SEP_LIMIT', SEP_LIMIT)
TOK_SEP_VCTR = Token('SEP_VCTR', SEP_VCTR)
TOK_SEP_MTRX = Token('SEP_MTRX', SEP_MTRX)
TOK_SEP_OR = Token('SEP_OR', SEP_OR)

TOK_SEP_TOK = Token('SEP_TOK', '')
TOK_SEP_PASS = Token('SEP_PASS', '')
TOK_NOVAR = Token('NOVAR', '')


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
	def kinds(self) -> list[str]:
		kinds=[]
		for n in self: kinds.extend(n.kinds)
		return kinds


# 1+2
class Term(Node):
	opr: Token = TOK_SEP_TOK


# (1+2 OR 3+4)
class Junc(Node):
	opr: Token = TOK_SEP_OR


# Value > (1+2 OR 3+4) : $var
class Limit(Node):
	opr: Token = TOK_SEP_PASS
	def check(self) -> 'Limit':
		if len(self)!=3: raise SyntaxError('E_NO_LIST')
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
	bindings = []
	mtrx, vctr = Matrix(), Vector()

	while tokens.peek():

		# LIMIT: Single axis constraint
		limit = Limit(Term(), Junc(), TOK_NOVAR)

		# LEFT
		if tokens.peek() == 'VAL': limit[LEFT].append(tokens.next())
		if tokens.peek() in MOD_KINDS:
			if not limit[LEFT]: limit[LEFT].append(Token('VAL', ELIDE))
			limit[LEFT].opr=tokens.next()
			if tokens.peek() not in DATUM_KINDS: raise SyntaxError('E_EXPR_DATUM')
			limit[LEFT].append(tokens.next())

		# CMP
		if tokens.peek() in CMP_KINDS:
			limit.opr=tokens.next()
			if tokens.peek() not in DATUM_KINDS: raise SyntaxError('E_TERM_DATUM')

		# RIGHT
		while tokens.peek() in DATUM_KINDS:
			if limit.opr.kind == 'SEP_PASS': limit.opr=Token('EQL', ELIDE)
			right_term = Term(tokens.next())
			if tokens.peek() in MOD_KINDS:
				right_term.opr=tokens.next()
				if tokens.peek() not in DATUM_KINDS: raise SyntaxError('E_EXPR_DATUM')
				right_term.append(tokens.next())
			limit[RIGHT].append(right_term.check())
			if tokens.peek() == 'SEP_OR': tokens.next()

		if limit[RIGHT] and not limit[LEFT]: limit[LEFT].append(Token('VAL', ELIDE))

		# ASSN
		if tokens.peek() == 'ASSN':
			if not limit[LEFT]: limit[LEFT].append(Token('VAL', ELIDE))
			tokens.next()
			if tokens.peek() != 'VAR': raise SyntaxError('E_ASSN_VAR')
			limit[AVAR] = tokens.next()
			bindings.append(limit[AVAR].lexeme)

		# FINAL LIMIT
		if limit[LEFT]:
			if len(mtrx)==0 and 'VSAME' in limit.kinds: raise SyntaxError('E_VSAME_OOB')			
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
			self[mtrx_idx].pad(Limit(
					Term(Token('VAL',ELIDE)).check(),
					Junc(Term(Token('VSAME',VSAME)).check()).check(),
					TOK_NOVAR,
					opr=Token('EQL',ELIDE)
				).check())

		self.results = [[[[] for limit in vctr] for vctr in mtrx] for mtrx in self]

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
		if kind=='PROB': return str(random.uniform(0, 1))

	@staticmethod
	def limit(bindings: List[str]|None = None) -> Memelang:
		if not bindings: bindings = []

		data: Memelang = ''

		comp = random.choice(['=','!=','>','<','<=','>='])

		# EMBEDDING
		if comp in {'<','<='} and random.randint(0, 2):
			data += VAL + '<=>' + Fuzz.datum('EMB') 
			if random.randint(0, 1): data += comp + Fuzz.datum('PROB')

		# FILTER
		else:
			# LEFT
			if random.randint(0, 1): data += VAL

			data+=comp

			# RIGHT
			if comp in {'=','!=','!'}:
				data_list_len = random.randint(1, 5)
				data_list: List[Any] = []
				for _ in range(data_list_len):
					datum_type = random.randint(1, 7)
					if datum_type == 1:  data_list.append(Fuzz.datum('QUOT'))
					elif datum_type == 2:  data_list.append(Fuzz.datum('FLOAT'))
					elif datum_type == 3 and bindings: data_list.append(random.choice(bindings))
					elif datum_type == 4 and VSAME in bindings: data_list.append(VSAME)
					else: data_list.append(Fuzz.datum('ALNUM'))
				data += SEP_OR.join(data_list)
			else: data += Fuzz.datum('FLOAT')

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
		return Fuzz.datum('ALNUM') + SEP_LIMIT + VAL + SEP_LIMIT + SEP_VCTR_PRETTY.join(Fuzz.datum('ALNUM') + SEP_LIMIT + Fuzz.limit() for _ in range(col_len)) + SEP_MTRX_PRETTY



### SQL ### 

'''
1. EXAMPLE QUERY
MEMELANG: roles _ actor "Mark Hamill",Mark ; movie _ ; rating >4 ;;
SQL COLS: SELECT actor, movie, rating FROM roles WHERE actor IN ('Mark Hamill', 'Mark') AND rating > 4;
SQL MEME: SELECT CONCAT_WS(' ', 'roles', t0.id, 'actor', t0.actor, ';', 'movie', t0.movie, ';', 'rating', t0.rating, ';;') AS meme FROM roles AS t0 WHERE t0.actor IN ('Mark Hamill', 'Mark') AND t0.rating > 4;

2. EXAMPLE JOIN
MEMELANG: roles _ actor "Mark Hamill" ; movie _ ; !@ @ @ ; actor _ ;;
SQL COLS: SELECT t0.id, t0.actor, t0.movie, t1.movie, t1.actor FROM roles AS t0, roles AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.id != t0.id AND t1.movie = t0.movie;
SQL MEME: SELECT CONCAT_WS(' ', 'roles', t0.id, 'actor', t0.actor, ';', 'movie', t0.movie, ';', t1.id, 'movie', t1.movie, ';', 'actor', t1.actor, ';;' ) AS meme FROM roles AS t0, roles AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.id != t0.id AND t1.movie = t0.movie;

3. EXAMPLE TABLE JOIN WHERE ACTOR NAME = MOVIE TITLE
MEMELANG: actors _ age >21; name _ ; roles _ title @ ;;
SQL COLS: SELECT t0.id, t0.name, t0.age, t1.title FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t1.title = t0.name;
SQL MEME: SELECT CONCAT_WS(' ', 'actors', t0.id, 'age', t0.age, ';', 'name', t0.name, ';', 'roles', t1.id, 'title', t1.title, ';;' ) AS meme FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t1.title = t0.name;

4. EXAMPLE EMBEDDING
MEMELANG: documents _ body <=>[0.1,0.2,0.3]>0.5 ; year >2005 ;;
SQL COLS: SELECT t0.id, t0.body<=>[0.1,0.2,0.3], t0.year from documents AS t0 WHERE t0.body<=>[0.1,0.2,0.3]>0.5 AND t0.year>2005;
SQL MEME: SELECT CONCAT_WS(' ', 'documents', t0.id, 'body', t0.body<=>[0.1,0.2,0.3], ';', 'year', t0.year, ';;') AS meme from documents AS t0 WHERE t0.body<=>[0.1,0.2,0.3]>0.5 AND t0.year>2005;
'''

SQL = str
Param = int|float|str|list

class SQLUtil():
	cmp2sql = {'EQL':'=','NOT':'!=','GT':'>','GE':'>=','LT':'<','LE':'<='}
	@staticmethod
	def escape(token: Token, bindings: dict) -> SQL:
		if token.kind == 'DBCOL': return token.datum
		elif token.kind == 'VAL': return SQLUtil.escape(bindings[VAL], bindings)
		elif token.kind == 'VSAME': return SQLUtil.escape(bindings[VSAME], bindings)
		elif token.kind == 'VAR':
			if token.lexeme not in bindings: raise SyntaxError('E_VAR_BIND')
			return SQLUtil.escape(bindings[token.lexeme], bindings)
		return '%s'

	@staticmethod
	def escape2(token: Token, bindings: dict) -> None|Param:
		if token.kind == 'DBCOL': return None
		elif token.kind == 'VAL': return SQLUtil.escape2(bindings[VAL], bindings)
		elif token.kind == 'VSAME': return SQLUtil.escape2(bindings[VSAME], bindings)
		elif token.kind == 'VAR':
			if token.lexeme not in bindings: raise SyntaxError('E_VAR_BIND')
			return SQLUtil.escape2(bindings[token.lexeme], bindings)
		return token.datum

	@staticmethod
	def select(term: Term, bindings: dict) -> Tuple[SQL, List[None|Param]]:
		sqlselect = SQLUtil.escape(term[0], bindings)
		sqlparams = [SQLUtil.escape2(term[0], bindings)]
		if term.opr.kind!='SEP_TOK':
			sqlselect += term.opr.lexeme + SQLUtil.escape(term[1], bindings)
			sqlparams.append(SQLUtil.escape2(term[1], bindings))

		return sqlselect, sqlparams

	@staticmethod
	def where(limit: Limit, bindings: dict) -> Tuple[SQL, List[None|Param]]:
		if limit.opr.kind == 'SEP_PASS': return '', []
		sym = SQLUtil.cmp2sql[limit.opr.kind]
		lp, rp = '', ''

		if len(limit[RIGHT]) > 1:
			lp, rp = '(', ')'
			if limit.opr.kind == 'EQL': sym = 'IN'
			elif limit.opr.kind == 'NOT': sym = 'NOT IN'
			else: raise SyntaxError()

		leftsql, params = SQLUtil.select(limit[LEFT], bindings)
		rights = []
		for right in limit[RIGHT]:
			sql, subparams = SQLUtil.select(right, bindings)
			rights.append(sql)
			params.extend(subparams)

		return f'{leftsql} {sym} {lp}'+ ', '.join(rights) + rp, params

	@staticmethod
	def deref(limit: Limit, bindings: dict) -> Tuple[bool, None|Token]:
		if limit.opr.kind != 'EQL': return False, None
		if len(limit[LEFT])>1: return False, None
		if len(limit[RIGHT])>1: return False, None
		if len(limit[RIGHT][0])>1: return False, None
		if limit[RIGHT][0][0].kind == 'VSAME' and bindings[VSAME]: return True, bindings[VSAME]
		return limit[RIGHT][0][0] == bindings[VSAME], limit[RIGHT][0][0]


class MemeSQLTable(Meme):
	output = 'meme'
	primary: str = 'id'

	def select(self) -> Tuple[SQL, List[Param]]:
		cte_idx: int = 0
		tbl_idx: int = 0
		sel_idx: int = 0
		sqlsels: Dict[int, SQL] = {}
		params: List[Param] = []
		axis_name: Dict[Axis, str] = {}
		name_axis: Dict[str, Axis] = {}
		
		for mtrx in self:
			cte_idx+=1
			froms, wheres, selectrows, orders, selectmemes, sel_params, whr_params, bindings = [], [], [], [], [], [], [], {}
			tbl_alias = None
			prev = {'val': None,'col': None, 'row': None, 'tbl': None}

			for vctr in mtrx:

				if not axis_name: # TO DO: MAKE THIS CHANGEABLE PER VCTR
					axis_name = {0: 'val', 1: 'col', 2: 'row', 3: 'tbl'}
					name_axis = {v: k for k, v in axis_name.items()}

				curr = {'val': None,'col': None, 'row': None, 'tbl': None}
				same = {'val': None,'col': None, 'row': None, 'tbl': None}
				
				for aname in ('col','row','tbl'):
					same[aname], curr[aname] = SQLUtil.deref(vctr[name_axis[aname]], {VSAME: prev[aname]})

				# JOIN
				if not same['tbl'] or not same['row']:

					# TABLE ALIAS
					if not curr['tbl'] or curr['tbl'].kind != 'ALNUM': raise SyntaxError('E_TBL_ALNUM')
					tbl_alias = f't{tbl_idx}'
					froms.append(f"{curr['tbl']} AS {tbl_alias}")
					tbl_idx += 1
					pricol = f"{tbl_alias}.{self.primary}"
					selectmemes.append(f"'{curr['tbl'].lexeme}'")

					# PRIMARY KEY
					bindings[VSAME]=prev['row'] if prev['row'] is not None else None
					curr['row']=bindings[VAL]=Token('DBCOL', pricol)
					where, param = SQLUtil.where(vctr[name_axis['row']], bindings)
					if where:
						wheres.append(where)
						whr_params.extend([p for p in param if p is not None])

					sel_idx+=1
					selectrows.append(f'{pricol} AS s{sel_idx}')
					selectmemes.append(pricol)

				if not curr['col'] or curr['col'].kind != 'ALNUM': raise SyntaxError('E_COL_ALNUM')

				col_name = curr['col'].datum
				col_alias = f"{tbl_alias}.{col_name}"
				sel_idx += 1
				sel_alias = f's{sel_idx}'

				if prev['val']: bindings[VSAME]=prev['val']
				curr['val']=bindings[VAL]=Token('DBCOL', col_alias)

				# SELECT
				select, param = SQLUtil.select(vctr[name_axis['val']][0], bindings)
				if select:
					selectrows.append(f'{select} AS {sel_alias}')
					selectmemes.extend([f"'{col_name}'", select, f"'{SEP_VCTR}'"])
					sel_params.extend([p for p in param if p is not None])

				# WHERE
				where, param = SQLUtil.where(vctr[name_axis['val']], bindings)
				if where:
					wheres.append(where)
					whr_params.extend([p for p in param if p is not None])

				for axis, aname in axis_name.items():
					if vctr[axis][AVAR].kind == 'VAR': bindings[vctr[axis][AVAR].lexeme]=curr[aname]

				prev = curr.copy()

			params.extend(sel_params+whr_params)
			selectmemes.append(f"'{SEP_MTRX}'")

			wherestr = '' if not wheres else ' WHERE ' + ' AND '.join(wheres)
			orderstr = '' if not orders else ' ORDER BY ' + ', '.join(orders)	

			if self.output=='meme': sqlsels[cte_idx] = f'SELECT CONCAT_WS(\'{SEP_LIMIT}\', ' + ', '.join(selectmemes) + f') AS meme FROM ' + ', '.join(froms) + wherestr + orderstr
			else: sqlsels[cte_idx] = 'SELECT ' + ', '.join(selectrows) + ' FROM ' + ', '.join(froms) + wherestr + orderstr

		return ' UNION ALL '.join(sqlsels[k] for k in sqlsels) +';', params
