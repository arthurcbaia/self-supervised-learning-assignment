# Guia de Métricas de Análise de Corpus

## Medidas Estatísticas

### 1. Distribuição de Palavras
**Visualização**: Boxplot de palavras por autor
- Mostra a distribuição total de palavras nos textos de cada autor
- A caixa mostra os quartis (percentis 25, 50 e 75)
- Outliers indicam textos excepcionalmente longos ou curtos
- Ajuda a identificar diferenças no comprimento típico dos textos entre autores

### 2. Comprimento Médio das Frases
**Visualização**: Gráfico de violino comparando Machado vs Outros
- Exibe a distribuição dos comprimentos das frases
- Seções mais largas representam comprimentos de frase mais comuns
- A forma mostra o padrão completo de distribuição
- Frases mais longas podem indicar um estilo de escrita mais complexo
- Frases mais curtas geralmente sugerem uma escrita mais direta ou simples

### 3. Riqueza Vocabular (Razão de Palavras Únicas)
**Visualização**: Boxplot da razão de palavras únicas
- Calculada como: número de palavras únicas / total de palavras
- Razões mais altas indicam vocabulário mais diverso
- Razões mais baixas sugerem uso mais repetitivo de palavras
- Ajuda a avaliar a amplitude vocabular e o estilo de escrita do autor
- Intervalos típicos: 0,3 a 0,6 (depende do comprimento do texto)

### 4. Densidade Lexical
**Visualização**: Boxplot da razão de palavras de conteúdo
- Mede a proporção de palavras de conteúdo (excluindo stopwords)
- Densidade mais alta sugere texto mais rico em informação
- Densidade mais baixa indica mais palavras funcionais/gramaticais
- Intervalos típicos: 0,4 a 0,6
- Importante para comparar estilos de escrita e complexidade

### 5. Comprimento Médio das Palavras
**Visualização**: Gráfico de violino da distribuição do comprimento das palavras
- Mostra o padrão de uso do comprimento das palavras
- Palavras em média mais longas frequentemente correlacionam com escrita mais formal ou técnica
- Comprimento médio menor pode indicar estilo mais acessível ou coloquial
- Em português, tipicamente varia de 4-6 letras por palavra em média

### 6. Percentual de Hapax Legomena
**Visualização**: Boxplot de palavras que aparecem apenas uma vez
- Mede a porcentagem de palavras usadas exatamente uma vez
- Percentual mais alto sugere vocabulário mais variado
- Percentual mais baixo indica uso mais repetitivo de palavras
- Intervalos típicos: 20% a 50% do total de palavras
- Importante para estudar riqueza vocabular e estilo de escrita

## Dicas de Interpretação

1. **Contexto Importa**: Considere o gênero e período ao interpretar resultados
2. **Tamanho da Amostra**: Textos maiores naturalmente mostrarão padrões diferentes dos menores
3. **Análise Combinada**: Observe múltiplas métricas em conjunto para melhores insights
4. **Contexto Histórico**: Lembre-se que o estilo de escrita de Machado de Assis reflete o português do século XIX

## Padrões Comuns

- Densidade lexical mais alta frequentemente correlaciona com escrita mais formal ou acadêmica
- Frases mais longas tipicamente indicam estruturas sintáticas mais complexas
- Percentual mais alto de hapax geralmente sugere uso de vocabulário mais sofisticado
- A razão de palavras únicas tende a diminuir conforme o comprimento do texto aumenta

## Usando Estas Informações

- Compare estilos de autores quantitativamente
- Acompanhe mudanças no estilo de escrita entre diferentes obras
- Identifique padrões característicos na escrita de um autor
- Apoie análise literária com dados quantitativos 