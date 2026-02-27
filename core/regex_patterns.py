import re

# --- PADRÕES DE TEXTO ---

# Palavras-chave para achar a tabela (Adicionei REFORÇO e ARMADURA baseada na sua imagem)
TABLE_HEADER_KEYWORDS = [
    "STEEL", "POS", "BIT", "NUMBER", "LENGTH", "RESUMO", "ACO", "AÇO", 
    "DIAM", "UNIT", "TOTAL", "QUANT", "REFORÇO", "REFORCO", "ARMADURA"
]

# REGEX DO AÇO (Robustez Mantida)
# Captura: "7 N1 ø10 C=200"
REGEX_DESENHO_ACO = r'(?P<qtd_txt>[\d\s]+[xX\*][\d\s]+|[\d]+)\s*(?:N|n)(?P<pos>\d+).*?(?:ø|%%c|diam|d|D)\s*(?P<bit>[\d\.,]+).*?(?:C=|c=|L=|l=)\s*(?P<comp>\d+)'

# REGEX DE ELEMENTO ESTRUTURAL (Vigas, Lajes, Pilares)
# Antes só aceitava V... agora aceita:
# V1, Viga 1, L1, Laje 2, P10, P 10, B5, LAJÃO
REGEX_TITULO_VIGA = r'(?P<nome>(?:V|L|P|B|Viga|Laje|Pilar|Bloco|Lajão|Lajao)\s*\d*)\s*(?:\([xX]?(?P<mult>\d+)\))?'

# Captura genérica de linha de dados
REGEX_TABLE_DATA_ROW = r'^[\w\d\.]+'