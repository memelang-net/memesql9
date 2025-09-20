'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for prompting LLMs

MEMELANG USES AXES
AXES ORDERED HIGH TO LOW
ALWAYS WHITESPACE MEANS "NEW AXIS"
NEVER SPACE AROUND OPERATOR
NEVER SPACE BETWEEN COMPARATOR/COMMA AND VALUES
NEVER SPACE BEFORE SEP_META
'''

MEMELANG_VER = 9.14

import random, re, json
from typing import List, Iterator, Iterable, Dict, Tuple, Union

Memelang = str

ELIDE = ''
SIGIL, VAL, MSAME, VSAME, SEP_META, SMLR, EOF =  '$', '_', '^', '@', ':', '~', None
SEP_AXIS, SEP_VCTR, SEP_MTRX, SEP_OR = ' ', ';', ';;', ','
SEP_VCTR_PRETTY, SEP_MTRX_PRETTY = ' ; ', ' ;;\n'
LEFT, RIGHT = 0, 1

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
	('SMLR',		re.escape(SMLR)),
	('VAL',			re.escape(VAL)),		# VALCARD, MATCHES WHOLE VALUE, NEVER QUOTE
	('MSAME',		re.escape(MSAME)),		# REFERENCES (MTRX-1, VCTR=-1, LIMIT)
	('VSAME',		re.escape(VSAME)),		# REFERENCES (MTRX,   VCTR-1,  LIMIT)
	('VAR',			r'\$[A-Za-z0-9_]+'),
	('ALNUM',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERICS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('SUB',			r'\-'), # AFTER INT/FLOAT
	('SEP_META',	re.escape(SEP_META)),
	('SEP_MTRX',	re.escape(SEP_MTRX)),
	('SEP_VCTR',	re.escape(SEP_VCTR)),
	('SEP_OR',		re.escape(SEP_OR)),
	('SEP_AXIS',	r'\s+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

CMP_KINDS = {'EQL':{'STR','NUM','DATA'},'NOT':{'STR','NUM','DATA'},'GT':{'NUM'},'GE':{'NUM'},'LT':{'NUM'},'LE':{'NUM'},'SMLR':{'STR'}}
MOD_KINDS = {'MUL':{'NUM'},'ADD':{'NUM'},'SUB':{'NUM'},'DIV':{'NUM'},'MOD':{'NUM'},'POW':{'NUM'},'L2':{'EMB'},'IP':{'EMB'},'COS':{'EMB'}} #,'TSQ':{'TSQ'}
DATUM_KINDS = {'ALNUM','QUOT','INT','FLOAT','VAR','VSAME','MSAME','VAL','EMB'}
IGNORE_KINDS = {'COMMENT','MTBL'}
META_KINDS = {'ALNUM','VAR'}

EBNF = '''
TERM ::= DATUM [MOD DATUM]
JUNC ::= {TERM} {SEP_OR {TERM}}
LIMIT ::= [TERM] [CMP] [JUNC]
AXIS ::= LIMIT {META VAR|ALNUM}
VCTR ::= AXIS {SEP_AXIS AXIS}
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
TOK_SEP_AXIS = Token('SEP_AXIS', SEP_AXIS)
TOK_SEP_VCTR = Token('SEP_VCTR', SEP_VCTR)
TOK_SEP_MTRX = Token('SEP_MTRX', SEP_MTRX)
TOK_SEP_OR = Token('SEP_OR', SEP_OR)
TOK_SEP_META = Token('SEP_META', SEP_META)
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
			elif diff<0: raise SyntaxError('E_FIRST_VECTOR_MUST_BE_LONGEST')

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


# Value > (1+2 OR 3+4)
class Limit(Node):
	opr: Token = TOK_SEP_PASS
	def check(self) -> 'Limit':
		if len(self)!=2: raise SyntaxError('E_NO_LIST')
		return self


# Limit : $var : meta
class Axis(Node):
	opr: Token = TOK_SEP_META
	def check(self) -> 'Node': 
		if len(self)==0: raise SyntaxError('E_NO_LIST')
		if not isinstance(self[0], Limit): raise SyntaxError('E_AXIS_LIMIT')
		return self


class Vector(Node):
	opr: Token = TOK_SEP_AXIS
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
	bindings: List[str] = []
	mtrx, vctr = Matrix(), Vector()

	while tokens.peek():

		# AXIS
		limit = Limit(Term(), Junc())
		axis = Axis(Limit())

		# LEFT
		if tokens.peek() == 'VAL': limit[LEFT].append(tokens.next())
		if tokens.peek() in MOD_KINDS: limit[LEFT]=parse_term(Token('VAL', ELIDE), tokens, bindings)
			
		# CMP
		if tokens.peek() in CMP_KINDS:
			limit.opr=tokens.next()
			if tokens.peek() not in DATUM_KINDS: raise SyntaxError('E_CMP_DATUM')

		# RIGHT
		while tokens.peek() in DATUM_KINDS:
			if limit.opr.kind == 'SEP_PASS': limit.opr=Token('EQL', ELIDE)
			limit[RIGHT].append(parse_term(tokens.next(), tokens, bindings))
			if tokens.peek() == 'SEP_OR':
				tokens.next()
				if tokens.peek() not in DATUM_KINDS: raise SyntaxError('E_OR_TRAIL')

		# META
		while tokens.peek() == 'SEP_META':
			tokens.next()
			t = tokens.next()
			if t.kind not in META_KINDS: raise SyntaxError('E_META_KIND')
			axis.append(t)
			if axis[-1].kind == 'VAR': bindings.append(t.lexeme)


		if (limit[RIGHT] or len(axis)>1) and not limit[LEFT]: limit[LEFT].append(Token('VAL', ELIDE))

		# FINAL LIMIT
		if limit[LEFT]:	
			axis[LEFT]=limit.check()
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
		if t.kind not in DATUM_KINDS: raise SyntaxError('E_EXPR_DATUM')
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
					Limit(
						Term(Token('VAL',ELIDE)).check(),
						Junc(Term(Token('VSAME',VSAME)).check()).check(),
						opr=Token('EQL',ELIDE)
					).check()
				)
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
		if kind=='PROB': return str(random.uniform(0, 1))

	@staticmethod
	def limit(bindings: List[str]|None = None) -> Memelang:
		if not bindings: bindings = []

		data: Memelang = ''

		comp = random.choice(['=','!=','>','<','<=','>='])

		# EMBEDDING
		if comp in {'<','<=','>','>='} and random.randint(0, 2):
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
				data_list: List[Memelang] = []
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
		return SEP_AXIS.join(vector) + SEP_VCTR_PRETTY

	@staticmethod
	def mtrx_table(col_len:int = 5) -> Memelang:
		return Fuzz.datum('ALNUM') + SEP_AXIS + VAL + SEP_AXIS + SEP_VCTR_PRETTY.join(Fuzz.datum('ALNUM') + SEP_AXIS + Fuzz.limit() for _ in range(col_len)) + SEP_MTRX_PRETTY



### SQL ### 

'''
1. EXAMPLE QUERY
MEMELANG: roles _ actor "Mark Hamill",Mark ; movie _ ; rating >4 ;;
SQL: SELECT t0.actor, t0.movie, t0.rating FROM roles as t0 WHERE (t0.actor = 'Mark Hamill' or t0.actor = 'Mark') AND t0.rating > 4;

2. EXAMPLE JOIN
MEMELANG: roles _ actor "Mark Hamill" ; movie _ ; !@ @ @ ; actor _ ;;
SQL: SELECT t0.id, t0.actor, t0.movie, t1.movie, t1.actor FROM roles AS t0, roles AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.id != t0.id AND t1.movie = t0.movie;

3. EXAMPLE TABLE JOIN WHERE ACTOR NAME = MOVIE TITLE
MEMELANG: actors _ age >21; name _ ; roles _ title @ ;;
MEMELANG(2): actors _ age >21; name _:$n ; roles _ title $n ;;
MEMELANG(3): actors _ age >21; name :$x ; roles _ title $x ;;
SQL: SELECT t0.id, t0.name, t0.age, t1.title FROM actors AS t0, roles AS t1 WHERE t0.age > 21 AND t1.title = t0.name;

4. EXAMPLE EMBEDDING
MEMELANG: movies _ description <=>[0.1,0.2,0.3]>0.5 ; year >2005 ;;
SQL: SELECT t0.id, t0.description<=>[0.1,0.2,0.3], t0.year from movies AS t0 WHERE t0.description<=>[0.1,0.2,0.3]>0.5 AND t0.year>2005;

5. EXAMPLE AGGREGATION
MEMELANG: roles _ rating:avg ; actor "Mark Hamill","Carrie Fisher":grp ;;
SQL: SELECT AVG(t0.rating), t0.actor FROM roles AS t0 WHERE (t0.actor = 'Mark Hamill' OR t0.actor = 'Carrie Fisher') GROUP BY t0.actor;

'''

SQL = str
Param = int|float|str|list
Agg = int
ANONE, ACNST, AGRP, AHAV = 0, 1, 2, 3

class SQLUtil():
	cmp2sql = {'EQL':'=','NOT':'!=','GT':'>','GE':'>=','LT':'<','LE':'<=','SMLR':'ILIKE'}
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
		limit = axis[LEFT]
		agg_func = {'sum': 'SUM', 'avg': 'AVG', 'min': 'MIN', 'max': 'MAX'}
		agg = ANONE
		sqlterm, sqlparams = SQLUtil.term(limit[LEFT], bindings)
		for t in axis[RIGHT:]:
			if t.lexeme in agg_func:
				if agg: raise SyntaxError('E_DBL_AGG')
				agg = AHAV
				sqlterm = agg_func[t.lexeme] + '(' + sqlterm + ')'
			elif t.lexeme == 'grp': agg = AGRP
			# TO DO: FLAG CONFLICTS
		return sqlterm, sqlparams, agg
		
	@staticmethod
	def where(axis: Axis, bindings: dict) -> Tuple[SQL, List[None|Param], Agg]:
		limit = axis[LEFT]
		if limit.opr.kind == 'SEP_PASS': return '', [], ANONE
		sym = SQLUtil.cmp2sql[limit.opr.kind]
		lp, rp, junc = '', '', ''

		if len(limit[RIGHT]) > 1:
			lp, rp = '(', ')'
			junc = 'AND' if limit.opr.kind == 'NOT' else 'OR'

		leftsql, params, agg = SQLUtil.select(limit, bindings)
		rights = []
		for right in limit[RIGHT]:
			sql, subparams = SQLUtil.term(right, bindings)

			if sym in ('LIKE','ILIKE'):
				sql = sql.replace('%s', "CONCAT('%', %s, '%')")

			rights.append(f"{leftsql} {sym} {sql}")
			params.extend(subparams)

		return lp + f' {junc} '.join(rights) + rp, params, agg

	@staticmethod
	def deref(limit: Limit, bindings: dict) -> Tuple[bool, None|Token]:
		if limit.opr.kind != 'EQL' or len(limit[LEFT])>1 or len(limit[RIGHT])!=1 or len(limit[RIGHT][0])!=1: return False, None
		if limit[RIGHT][0][0].kind == 'VSAME': return True, bindings.get(VSAME)
		return limit[RIGHT][0][0] == bindings.get(VSAME), limit[RIGHT][0][0]


class MemeSQLTable(Meme):
	def select(self) -> List[Tuple[SQL, List[Param]]]:
		tbl_idx: int = 0
		sqls: list[Tuple[SQL, List[Param]]] = []
		axis_name: Dict[int, str] = {}
		name_axis: Dict[str, int] = {}
		
		for mtrx in self:
			selectall = False
			tbl_alias = None
			froms, wheres, selects, orderbys, groupbys, havings, bindings = [], [], [], [], [], [], {}
			prev = {'val': None,'col': None, 'row': None, 'tbl': None}

			for vctr in mtrx:

				if len(vctr)!=4: raise SyntaxError('E_SQL_VCTR_LEN')

				if not axis_name: # TO DO: MAKE THIS CHANGEABLE PER VCTR
					primary = 'id'
					axis_name = {0: 'val', 1: 'col', 2: 'row', 3: 'tbl'}
					name_axis = {v: k for k, v in axis_name.items()}

				curr = {'val': None,'col': None, 'row': None, 'tbl': None}
				same = {'val': None,'col': None, 'row': None, 'tbl': None}
				
				for aname in ('col','row','tbl'):
					same[aname], curr[aname] = SQLUtil.deref(vctr[name_axis[aname]][LEFT], {VSAME: prev[aname]})

				# JOIN
				if not same['tbl'] or not same['row']:

					if selectall: selects.append((f'{tbl_alias}.*', [], ANONE))
					selectall = False

					# TBL
					if not curr['tbl'] or curr['tbl'].kind != 'ALNUM': raise SyntaxError('E_TBL_ALNUM')
					tbl_alias = f't{tbl_idx}'
					froms.append(f"{curr['tbl']} AS {tbl_alias}")
					tbl_idx += 1
					pricol = f"{tbl_alias}.{primary}"

					# ROW
					bindings[VSAME]=prev['row'] if prev['row'] is not None else None
					curr['row']=bindings[VAL]=Token('DBCOL', pricol)
					where, param, _ = SQLUtil.where(vctr[name_axis['row']], bindings)
					if where: wheres.append((where, param, ANONE))

					selects.extend([(f"'{curr['tbl'].lexeme}' AS _a3", [], ACNST), (f"{pricol} AS _a2", [], ANONE)])

				# COL
				if not curr['col']: raise SyntaxError('E_COL')
				elif curr['col'].kind == 'VAL' and vctr[name_axis['val']][LEFT][LEFT][LEFT].kind == 'VAL': # LEFT CHECK IS LAZY
					selectall=True
					continue
				elif curr['col'].kind != 'ALNUM': raise SyntaxError('E_COL_ALNUM')

				col_name = curr['col'].datum
				col_alias = f"{tbl_alias}.{col_name}"

				# VAL
				if prev['val']: bindings[VSAME]=prev['val']
				curr['val']=bindings[VAL]=Token('DBCOL', col_alias)

				valaxis = vctr[name_axis['val']]
				select = SQLUtil.select(valaxis, bindings)
				selects.append(select)

				if any(t.lexeme=='grp' for t in valaxis[RIGHT:]): groupbys.append(select)
				if any(t.lexeme=='asc' for t in valaxis[RIGHT:]): orderbys.append((select[0]+' ASC', select[1], select[2]))
				elif any(t.lexeme=='dsc' for t in valaxis[RIGHT:]): orderbys.append((select[0]+' DESC', select[1], select[2]))

				where = SQLUtil.where(valaxis, bindings)
				if where[0]:
					if where[2]==AHAV: havings.append(where)
					else: wheres.append(where)
			
				for axis, aname in axis_name.items():
					for t in vctr[axis][RIGHT]:
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
			params = [p for s in selects+wheres+groupbys+havings+orderbys for p in s[1] if p is not None]
			sqls.append((selectstr + fromstr + wherestr + groupbystr + havingstr + orderbystr, params))

		return sqls
