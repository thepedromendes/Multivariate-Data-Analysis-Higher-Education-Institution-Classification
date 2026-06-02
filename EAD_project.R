# ============================================================
# PROJECTO EAD — ANÁLISE MULTIVARIADA
# LDA, QDA, Regressão Logística, KNN, PCA e Clustering
# Autora: Margarida Pessoa
# ============================================================

packages <- c(
  "tidyverse", "janitor", "caret", "MASS", "pROC",
  "factoextra", "cluster", "corrplot", "moments",
  "biotools", "MVN", "broom"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
invisible(lapply(packages, install_if_missing))
invisible(lapply(packages, library, character.only = TRUE))

options(scipen = 999)
set.seed(2304093)

# ============================================================
# WORKING DIRECTORY
# ============================================================
# Opção A (recomendada): Session → Set Working Directory →
#                        To Source File Location
#
# Opção B: descomenta e ajusta:
# setwd("/Users/margaridapessoa/Desktop/FCUP/EAD/2nd Part Cluster Analysis/Computational Project")

if (!file.exists("Data/volis_dataset.csv")) {
  stop(
    "\nDataset não encontrado em: ", getwd(), "/Data/volis_dataset.csv",
    "\n\nSoluções:\n",
    "  1. Session → Set Working Directory → To Source File Location\n",
    "  2. Descomenta e ajusta o setwd() acima\n"
  )
}

dir.create("figures",  showWarnings = FALSE)
dir.create("outputs",  showWarnings = FALSE)

# ============================================================
# 1. IMPORTAR DADOS
# ============================================================

dados <- readr::read_csv(
  "Data/volis_dataset.csv",
  show_col_types = FALSE
) |> janitor::clean_names()

dados <- dados |>
  dplyr::mutate(
    bl_private = factor(
      ifelse(as.character(bl_private) %in% c("TRUE", "true", "1", "Private"),
             "Private", "Public"),
      levels = c("Public", "Private")
    )
  )

cat("\nDimensão inicial do dataset:", nrow(dados), "×", ncol(dados), "\n")
cat("\nDistribuição da variável alvo:\n")
print(table(dados$bl_private))
cat(sprintf("  (%.1f%% Private | %.1f%% Public)\n",
            100 * mean(dados$bl_private == "Private"),
            100 * mean(dados$bl_private == "Public")))

# ============================================================
# 2. VARIÁVEIS DERIVADAS
# ============================================================
# Transformar contagens em taxas evita o "efeito dimensão":
# instituições maiores dominam distâncias apenas por volume.

dados <- dados |>
  dplyr::mutate(
    tx_acceptance_rate    = qt_applications_accepted / qt_applications_received,
    tx_yield_rate         = qt_students_enrolled     / qt_applications_accepted,
    tx_postgrad_undergrad = qt_postgraduate_students / qt_undergraduate_students,
    vl_total_cost_proxy   = vl_tuition_outstate + vl_room_board +
      vl_books_cost + vl_personal_expenses
  ) |>
  dplyr::mutate(
    dplyr::across(where(is.numeric), ~ ifelse(is.infinite(.x), NA, .x))
  )

# ============================================================
# 3. CORRECÇÃO DE OUTLIERS / VALORES INCONSISTENTES
# ============================================================
# Tuition > 100 000 USD: valor impossível (Harvard custa ~55k) → NA
# Yield e acceptance > 1: matematicamente impossível → NA

cat("\nResumo ANTES da correcção de outliers:\n")
print(summary(dados[, c("vl_tuition_outstate", "tx_yield_rate", "tx_acceptance_rate")]))

dados <- dados |>
  dplyr::mutate(
    vl_tuition_outstate = ifelse(vl_tuition_outstate > 100000, NA, vl_tuition_outstate),
    tx_yield_rate       = ifelse(tx_yield_rate > 1,            NA, tx_yield_rate),
    tx_acceptance_rate  = ifelse(tx_acceptance_rate > 1,       NA, tx_acceptance_rate),
    # Recalcular custo total com tuition corrigido
    vl_total_cost_proxy = vl_tuition_outstate + vl_room_board +
      vl_books_cost + vl_personal_expenses
  )

cat("\nResumo DEPOIS da correcção de outliers:\n")
print(summary(dados[, c("vl_tuition_outstate", "tx_yield_rate", "tx_acceptance_rate")]))

# ============================================================
# 4. TRANSFORMAÇÕES LOGARÍTMICAS
# ============================================================
# Variáveis com assimetria positiva grave (|skew| > 2):
# log1p(x) = log(1+x) — seguro para x = 0.

dados <- dados |>
  dplyr::mutate(
    log_applications_received = log1p(qt_applications_received),
    log_students_enrolled     = log1p(qt_students_enrolled),
    log_tuition_outstate      = log1p(vl_tuition_outstate)
  )

# ============================================================
# 5. VALORES AUSENTES
# ============================================================

num_vars <- dados |> dplyr::select(where(is.numeric)) |> names()

missing_table <- dados |>
  dplyr::summarise(dplyr::across(everything(), ~ sum(is.na(.x)))) |>
  tidyr::pivot_longer(everything(), names_to = "variavel", values_to = "n_missing") |>
  dplyr::mutate(pct = round(100 * n_missing / nrow(dados), 2)) |>
  dplyr::filter(n_missing > 0) |>
  dplyr::arrange(desc(n_missing))

cat("\nVariáveis com valores ausentes:\n")
print(missing_table)
readr::write_csv(missing_table, "outputs/missing_values.csv")

# ============================================================
# 6. ESTATÍSTICAS DESCRITIVAS
# ============================================================
# Indicadores de posição, dispersão e forma — exigidos pelas guidelines.

desc_table <- dados |>
  dplyr::select(dplyr::all_of(num_vars)) |>
  dplyr::summarise(dplyr::across(everything(), list(
    n        = ~ sum(!is.na(.x)),
    mean     = ~ round(mean(.x,   na.rm = TRUE), 3),
    median   = ~ round(median(.x, na.rm = TRUE), 3),
    sd       = ~ round(sd(.x,     na.rm = TRUE), 3),
    cv       = ~ round(sd(.x, na.rm=TRUE) / abs(mean(.x, na.rm=TRUE)), 3),
    min      = ~ round(min(.x,    na.rm = TRUE), 3),
    q1       = ~ round(quantile(.x, 0.25, na.rm = TRUE), 3),
    q3       = ~ round(quantile(.x, 0.75, na.rm = TRUE), 3),
    max      = ~ round(max(.x,    na.rm = TRUE), 3),
    iqr      = ~ round(IQR(.x,    na.rm = TRUE), 3),
    skewness = ~ round(moments::skewness(.x, na.rm = TRUE), 3),
    kurtosis = ~ round(moments::kurtosis(.x, na.rm = TRUE), 3)
  ), .names = "{.col}__{.fn}")) |>
  tidyr::pivot_longer(everything(),
                      names_to  = c("variavel", ".value"),
                      names_sep = "__")

cat("\nEstatísticas descritivas completas:\n")
print(desc_table, n = Inf)
readr::write_csv(desc_table, "outputs/descriptive_stats.csv")

cat("\nVariáveis com assimetria positiva grave (skewness > 2):\n")
print(desc_table |> dplyr::filter(skewness > 2) |> dplyr::pull(variavel))

# ============================================================
# 7. GRÁFICOS EXPLORATÓRIOS
# ============================================================

vars_plot <- intersect(c(
  "log_applications_received", "log_students_enrolled",
  "log_tuition_outstate", "vl_room_board",
  "pc_faculty_with_phd", "vl_student_faculty_ratio",
  "pc_alumni_donors", "pc_graduation_rate",
  "tx_acceptance_rate", "tx_yield_rate"
), num_vars)

cores <- c("Public" = "#2196F3", "Private" = "#F44336")

dados_long <- dados |>
  dplyr::select(dplyr::all_of(vars_plot), bl_private) |>
  tidyr::pivot_longer(cols = -bl_private)

# Density plots
p1 <- ggplot2::ggplot(dados_long,
                      ggplot2::aes(x = value, fill = bl_private, colour = bl_private)) +
  ggplot2::geom_density(alpha = 0.4, na.rm = TRUE) +
  ggplot2::facet_wrap(~ name, scales = "free", ncol = 3) +
  ggplot2::scale_fill_manual(values = cores) +
  ggplot2::scale_colour_manual(values = cores) +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Distribuição das variáveis por tipo de instituição",
                x = NULL, y = "Densidade", fill = "Tipo", colour = "Tipo")
print(p1)
ggplot2::ggsave("figures/distribuicoes.png", p1, width = 12, height = 8, dpi = 300)

# Boxplots
p2 <- ggplot2::ggplot(dados_long,
                      ggplot2::aes(x = bl_private, y = value, fill = bl_private)) +
  ggplot2::geom_boxplot(na.rm = TRUE, outlier.size = 0.6, outlier.alpha = 0.5) +
  ggplot2::facet_wrap(~ name, scales = "free_y", ncol = 3) +
  ggplot2::scale_fill_manual(values = cores) +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Boxplots por tipo de instituição",
                x = NULL, y = NULL, fill = "Tipo")
print(p2)
ggplot2::ggsave("figures/boxplots.png", p2, width = 12, height = 8, dpi = 300)

# Histogramas
p_hist <- ggplot2::ggplot(dados_long,
                          ggplot2::aes(x = value, fill = bl_private)) +
  ggplot2::geom_histogram(bins = 30, alpha = 0.6, na.rm = TRUE,
                          position = "identity") +
  ggplot2::facet_wrap(~ name, scales = "free", ncol = 3) +
  ggplot2::scale_fill_manual(values = cores) +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Histogramas por tipo de instituição",
                x = NULL, y = "Frequência", fill = "Tipo")
print(p_hist)
ggplot2::ggsave("figures/histogramas.png", p_hist, width = 12, height = 8, dpi = 300)

# Violin plots
p_violin <- ggplot2::ggplot(dados_long,
                            ggplot2::aes(x = bl_private, y = value, fill = bl_private)) +
  ggplot2::geom_violin(alpha = 0.5, na.rm = TRUE) +
  ggplot2::geom_boxplot(width = 0.12, outlier.size = 0.3, na.rm = TRUE) +
  ggplot2::facet_wrap(~ name, scales = "free_y", ncol = 3) +
  ggplot2::scale_fill_manual(values = cores) +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Violin plots por tipo de instituição",
                x = NULL, y = NULL, fill = "Tipo")
print(p_violin)
ggplot2::ggsave("figures/violin_plots.png", p_violin, width = 12, height = 8, dpi = 300)

# Scatter 1: log tuition vs graduation rate
p_scatter1 <- ggplot2::ggplot(dados,
                              ggplot2::aes(x = log_tuition_outstate,
                                           y = pc_graduation_rate,
                                           colour = bl_private)) +
  ggplot2::geom_point(alpha = 0.7, na.rm = TRUE) +
  ggplot2::geom_smooth(method = "lm", se = FALSE, na.rm = TRUE) +
  ggplot2::scale_colour_manual(values = cores) +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Log Tuition vs Graduation Rate",
                x = "log(1 + Tuition Outstate)", y = "Graduation Rate (%)", colour = "Tipo")
print(p_scatter1)
ggplot2::ggsave("figures/scatter_tuition_graduation.png", p_scatter1,
                width = 8, height = 6, dpi = 300)

# Scatter 2: acceptance rate vs yield rate
p_scatter2 <- ggplot2::ggplot(dados,
                              ggplot2::aes(x = tx_acceptance_rate,
                                           y = tx_yield_rate,
                                           colour = bl_private)) +
  ggplot2::geom_point(alpha = 0.7, na.rm = TRUE) +
  ggplot2::geom_smooth(method = "lm", se = FALSE, na.rm = TRUE) +
  ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  ggplot2::scale_colour_manual(values = cores) +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Acceptance Rate vs Yield Rate",
                x = "Acceptance Rate", y = "Yield Rate", colour = "Tipo")
print(p_scatter2)
ggplot2::ggsave("figures/scatter_acceptance_yield.png", p_scatter2,
                width = 8, height = 6, dpi = 300)

# ============================================================
# 8. ANÁLISE BIVARIADA — TESTES DE COMPARAÇÃO DE GRUPOS
# ============================================================
# Welch t-test (robusto a variâncias diferentes) + Wilcoxon (não-paramétrico).
# Ambos testam H₀: distribuição igual entre Public e Private.

cat("\n--- TESTES BIVARIADOS (t-test Welch + Wilcoxon) ---\n")

biv_tests <- purrr::map_dfr(num_vars, function(v) {
  form <- as.formula(paste(v, "~ bl_private"))
  tt   <- tryCatch(t.test(form, data = dados), error = function(e) NULL)
  wt   <- tryCatch(wilcox.test(form, data = dados, exact = FALSE),
                   error = function(e) NULL)
  dplyr::tibble(
    variavel       = v,
    mean_public    = ifelse(is.null(tt), NA, round(tt$estimate[1], 3)),
    mean_private   = ifelse(is.null(tt), NA, round(tt$estimate[2], 3)),
    t_p_value      = ifelse(is.null(tt), NA, round(tt$p.value, 6)),
    wilcox_p_value = ifelse(is.null(wt), NA, round(wt$p.value, 6)),
    significativa  = ifelse(is.null(tt), NA,
                            ifelse(tt$p.value < 0.05, "Sim ***", "Não"))
  )
}) |> dplyr::arrange(t_p_value)

print(biv_tests, n = Inf)
readr::write_csv(biv_tests, "outputs/bivariate_tests.csv")

cat("\nVariáveis NÃO significativas (p > 0.05):\n")
ns <- biv_tests |> dplyr::filter(t_p_value > 0.05) |> dplyr::pull(variavel)
if (length(ns) == 0) cat("Todas as variáveis são significativas.\n") else print(ns)

# ============================================================
# 9. CORRELAÇÃO
# ============================================================
# As versões originais de apps, enrolled e tuition são removidas
# pois foram substituídas pelas versões log (menos assimétricas).

num_data <- dados |>
  dplyr::select(where(is.numeric)) |>
  dplyr::select(
    -qt_applications_received,
    -qt_students_enrolled,
    -vl_tuition_outstate
  ) |>
  dplyr::mutate(dplyr::across(everything(),
                              ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)))

cor_mat <- cor(num_data, use = "pairwise.complete.obs")

cat("\nPares com correlação forte (|r| > 0.70):\n")
cor_forte <- as.data.frame(as.table(cor_mat)) |>
  dplyr::filter(Var1 != Var2, abs(Freq) > 0.70) |>
  dplyr::arrange(desc(abs(Freq))) |>
  dplyr::distinct(Freq, .keep_all = TRUE)
print(cor_forte)

png("figures/correlacao.png", width = 1600, height = 1400, res = 200)
corrplot::corrplot(cor_mat,
                   method = "color", type = "upper", order = "hclust",
                   tl.cex = 0.5, tl.col = "black",
                   addCoef.col = "black", number.cex = 0.4)
dev.off()
cat("Correlograma guardado em figures/correlacao.png\n")

# ============================================================
# 10. PCA
# ============================================================
# O target NÃO é usado para construir o PCA.
# É usado apenas a posteriori para colorir os indivíduos.

pca_res <- prcomp(num_data, center = TRUE, scale. = TRUE)

pca_var_df <- data.frame(
  PC       = paste0("PC", seq_along(pca_res$sdev)),
  Eigenval = round(pca_res$sdev^2, 3),
  VarExpl  = round(pca_res$sdev^2 / sum(pca_res$sdev^2) * 100, 2),
  CumVar   = round(cumsum(pca_res$sdev^2 / sum(pca_res$sdev^2)) * 100, 2)
)
cat("\nVariância explicada pelo PCA:\n")
print(head(pca_var_df, 10))

n_pcs_80 <- which(pca_var_df$CumVar >= 80)[1]
cat(sprintf("→ %d PCs explicam ≥ 80%% da variância total.\n", n_pcs_80))
readr::write_csv(pca_var_df, "outputs/pca_variance.csv")

# Loadings PC1 e PC2
loadings_df <- as.data.frame(pca_res$rotation[, 1:4]) |>
  tibble::rownames_to_column("variavel") |>
  dplyr::arrange(desc(abs(PC1)))
cat("\nTop 10 loadings PC1:\n")
print(head(loadings_df, 10))
readr::write_csv(loadings_df, "outputs/pca_loadings.csv")

# Scree plot
p_pca1 <- factoextra::fviz_eig(pca_res, addlabels = TRUE) +
  ggplot2::labs(title = "Scree Plot — PCA") + ggplot2::theme_bw()
print(p_pca1)
ggplot2::ggsave("figures/pca_screeplot.png", p_pca1, width = 8, height = 6, dpi = 300)

# PCA indivíduos
p_pca2 <- factoextra::fviz_pca_ind(
  pca_res, habillage = dados$bl_private,
  addEllipses = TRUE, ellipse.type = "confidence",
  geom = "point", repel = FALSE, alpha.ind = 0.6
) +
  ggplot2::labs(title = "PCA por tipo de instituição") + ggplot2::theme_bw()
print(p_pca2)
ggplot2::ggsave("figures/pca_individuos.png", p_pca2, width = 8, height = 6, dpi = 300)

# PCA variáveis
p_pca3 <- factoextra::fviz_pca_var(
  pca_res, repel = TRUE, labelsize = 3, col.var = "contrib",
  gradient.cols = c("#2c7bb6", "#ffffbf", "#d7191c")
) +
  ggplot2::labs(title = "Contribuição das variáveis no PCA") + ggplot2::theme_bw()
print(p_pca3)
ggplot2::ggsave("figures/pca_variaveis.png", p_pca3, width = 9, height = 7, dpi = 300)

# ============================================================
# 10B. DATASET PARA CLUSTERING E MODELAÇÃO
# ============================================================
# Remove: identificador textual, variáveis originais substituídas por log,
# e vl_total_cost_proxy (soma das componentes já presentes).

dados_modelo <- dados |>
  dplyr::select(
    -nm_college,
    -vl_total_cost_proxy,
    -qt_applications_received,
    -qt_students_enrolled,
    -vl_tuition_outstate
  ) |>
  dplyr::mutate(dplyr::across(where(is.numeric),
                              ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)))

num_model_data   <- dados_modelo |> dplyr::select(where(is.numeric))
scaled_model_data <- scale(num_model_data)

# ============================================================
# 11. CLUSTERING (K-MEANS)
# ============================================================

p_elbow <- factoextra::fviz_nbclust(scaled_model_data, kmeans,
                                    method = "wss", k.max = 8) +
  ggplot2::theme_bw() + ggplot2::labs(title = "Elbow Method")
print(p_elbow)
ggplot2::ggsave("figures/kmeans_elbow.png", p_elbow, width = 7, height = 5, dpi = 300)

p_sil <- factoextra::fviz_nbclust(scaled_model_data, kmeans,
                                  method = "silhouette", k.max = 8) +
  ggplot2::theme_bw() + ggplot2::labs(title = "Silhouette Method")
print(p_sil)
ggplot2::ggsave("figures/kmeans_silhouette.png", p_sil, width = 7, height = 5, dpi = 300)

# k = 2: alinhado com a variável alvo binária
set.seed(2304093)
km2 <- kmeans(scaled_model_data, centers = 2, nstart = 25)
cat("\nTabela Clusters k=2 vs Tipo de Instituição (target não usado no clustering):\n")
print(table(Cluster = km2$cluster, Tipo = dados_modelo$bl_private))

p_cluster2 <- factoextra::fviz_cluster(km2, data = scaled_model_data, geom = "point") +
  ggplot2::labs(title = "K-Means com k = 2") + ggplot2::theme_bw()
print(p_cluster2)
ggplot2::ggsave("figures/kmeans_clusters_k2.png", p_cluster2, width = 8, height = 6, dpi = 300)

# k = 5: sugerido pela silhouette
set.seed(2304093)
km5 <- kmeans(scaled_model_data, centers = 5, nstart = 25)
cat("\nTabela Clusters k=5 vs Tipo de Instituição:\n")
print(table(Cluster = km5$cluster, Tipo = dados_modelo$bl_private))

p_cluster5 <- factoextra::fviz_cluster(km5, data = scaled_model_data, geom = "point") +
  ggplot2::labs(title = "K-Means com k = 5") + ggplot2::theme_bw()
print(p_cluster5)
ggplot2::ggsave("figures/kmeans_clusters_k5.png", p_cluster5, width = 8, height = 6, dpi = 300)

# ============================================================
# 12. VERIFICAÇÃO DE PRESSUPOSTOS (LDA / QDA)
# ============================================================

# --- 12.1 Normalidade multivariada por grupo (Mardia) ---
# Aplicado sobre os primeiros 5 PCs (reduz colinearidade).
cat("\n--- Teste de Mardia (normalidade multivariada por grupo) ---\n")
for (g in levels(dados_modelo$bl_private)) {
  temp     <- dados_modelo |> dplyr::filter(bl_private == g) |>
    dplyr::select(where(is.numeric))
  pca_temp <- prcomp(temp, center = TRUE, scale. = TRUE)
  pcs      <- as.data.frame(pca_temp$x[, 1:min(5, ncol(pca_temp$x))])
  cat(sprintf("\nGrupo %s (n = %d):\n", g, nrow(pcs)))
  res_mvn <- tryCatch(MVN::mvn(pcs, mvnTest = "mardia", multivariatePlot = "none"),
                      error = function(e) { cat("  Erro:", e$message, "\n"); NULL })
  if (!is.null(res_mvn)) print(res_mvn$multivariateNormality)
}

# --- 12.2 Homocedasticidade — Box's M ---
# H₀: Σ_Public = Σ_Private.
# Rejeitar H₀ → QDA teoricamente mais adequado que LDA.
cat("\n--- Box's M Test (H₀: Σ_Public = Σ_Private) ---\n")
boxm_result <- tryCatch(
  biotools::boxM(dados_modelo |> dplyr::select(where(is.numeric)),
                 dados_modelo$bl_private),
  error = function(e) { cat("Erro:", e$message, "\n"); NULL }
)
if (!is.null(boxm_result)) {
  print(boxm_result)
  cat(ifelse(boxm_result$p.value < 0.05,
             "→ Rejeita H₀: matrizes de covariância diferentes. QDA adequado.\n",
             "→ Não rejeita H₀: matrizes iguais. LDA adequado.\n"))
}

# ============================================================
# 13. PREPARAÇÃO PARA MODELOS SUPERVISIONADOS
# ============================================================

set.seed(2304093)
idx        <- caret::createDataPartition(dados_modelo$bl_private, p = 0.7, list = FALSE)
train_data <- dados_modelo[ idx, ]
test_data  <- dados_modelo[-idx, ]

cat(sprintf("\nTreino: %d obs. | Teste: %d obs.\n", nrow(train_data), nrow(test_data)))
cat("Distribuição target (treino):\n")
print(round(prop.table(table(train_data$bl_private)) * 100, 1))

ctrl <- caret::trainControl(
  method          = "cv",
  number          = 10,
  classProbs      = TRUE,
  summaryFunction = caret::twoClassSummary,
  savePredictions = "final"
)

formula_model <- bl_private ~ .

# ============================================================
# 14. LDA
# ============================================================
# Fronteira de decisão linear; assume Σ_Public = Σ_Private.

set.seed(2304093)
fit_lda <- caret::train(
  formula_model, data = train_data,
  method = "lda", preProcess = c("center", "scale"),
  trControl = ctrl, metric = "ROC"
)
cat("\nLDA — AUC CV:", round(max(fit_lda$results$ROC), 4), "\n")

# Coeficientes LD1 padronizados (contribuição de cada variável)
# apply() apenas sobre colunas numéricas — bl_private (factor) excluído
lda_mass      <- MASS::lda(formula_model, data = train_data)
train_numeric <- train_data |> dplyr::select(where(is.numeric))
sds           <- apply(train_numeric, 2, sd)
ld1_raw       <- lda_mass$scaling[, 1]
ld1_std       <- ld1_raw * sds[names(ld1_raw)]
ld1_df        <- data.frame(variavel = names(ld1_std),
                            coef_LD1 = round(ld1_std, 4)) |>
  dplyr::arrange(desc(abs(coef_LD1)))
cat("\nCoeficientes padronizados LD1 (Top 10):\n")
print(head(ld1_df, 10))
readr::write_csv(ld1_df, "outputs/lda_coefs_ld1.csv")

p_ld1 <- ggplot2::ggplot(head(ld1_df, 15),
                         ggplot2::aes(x = reorder(variavel, abs(coef_LD1)),
                                      y = coef_LD1,
                                      fill = coef_LD1 > 0)) +
  ggplot2::geom_col() + ggplot2::coord_flip() +
  ggplot2::scale_fill_manual(values = c("TRUE" = "#F44336", "FALSE" = "#2196F3"),
                             labels = c("TRUE" = "→ Private", "FALSE" = "→ Public")) +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "LDA: Coeficientes padronizados LD1",
                x = NULL, y = "Coeficiente LD1 padronizado", fill = "Direcção")
print(p_ld1)
ggplot2::ggsave("figures/lda_coefs_ld1.png", p_ld1, width = 8, height = 6, dpi = 300)

# ============================================================
# 15. QDA
# ============================================================
# Fronteira quadrática; estima Σk separadamente por classe.

set.seed(2304093)
fit_qda <- caret::train(
  formula_model, data = train_data,
  method = "qda", preProcess = c("center", "scale"),
  trControl = ctrl, metric = "ROC"
)
cat("QDA — AUC CV:", round(max(fit_qda$results$ROC), 4), "\n")

# ============================================================
# 16. REGRESSÃO LOGÍSTICA
# ============================================================
# Não assume normalidade; coeficientes interpretáveis como log-odds.
# exp(β) = odds ratio: aumento multiplicativo nas odds de ser Private.

set.seed(2304093)
fit_logit <- caret::train(
  formula_model, data = train_data,
  method = "glm", family = binomial,
  preProcess = c("center", "scale"),
  trControl = ctrl, metric = "ROC"
)
cat("LR  — AUC CV:", round(max(fit_logit$results$ROC), 4), "\n")

# Odds ratios com IC 95%
logit_coef <- broom::tidy(fit_logit$finalModel) |>
  dplyr::filter(term != "(Intercept)") |>
  dplyr::mutate(
    OR      = round(exp(estimate), 4),
    IC_low  = round(exp(estimate - 1.96 * std.error), 4),
    IC_high = round(exp(estimate + 1.96 * std.error), 4),
    p.value = round(p.value, 5)
  ) |>
  dplyr::arrange(p.value)

cat("\nOdds Ratios — Regressão Logística (Top 10 por significância):\n")
print(logit_coef |> dplyr::select(term, OR, IC_low, IC_high, p.value) |> head(10))
readr::write_csv(logit_coef, "outputs/logistic_odds_ratios.csv")

# Forest plot de OR
p_or <- logit_coef |>
  dplyr::filter(p.value < 0.05) |>
  ggplot2::ggplot(ggplot2::aes(x = reorder(term, OR), y = OR)) +
  ggplot2::geom_point(colour = "#D85A30", size = 2) +
  ggplot2::geom_errorbar(ggplot2::aes(ymin = IC_low, ymax = IC_high),
                         width = 0.3, colour = "#D85A30") +
  ggplot2::geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50") +
  ggplot2::coord_flip() +
  ggplot2::scale_y_log10() +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Odds Ratios — Regressão Logística (p < 0.05)",
                subtitle = "IC 95% | escala logarítmica | OR = 1: sem efeito",
                x = NULL, y = "Odds Ratio (log scale)")
print(p_or)
ggplot2::ggsave("figures/logistic_odds_ratios.png", p_or, width = 9, height = 7, dpi = 300)

# ============================================================
# 17. KNN
# ============================================================
# Não-paramétrico; K óptimo por AUC-ROC em 10-fold CV.
# Padronização já feita via preProcess — obrigatória para distâncias Euclidianas.

set.seed(2304093)
fit_knn <- caret::train(
  formula_model, data = train_data,
  method = "knn",
  tuneGrid = expand.grid(k = seq(1, 25, by = 2)),
  preProcess = c("center", "scale"),
  trControl = ctrl, metric = "ROC"
)
best_k <- fit_knn$bestTune$k
cat(sprintf("KNN — K óptimo: %d | AUC CV: %.4f\n", best_k, max(fit_knn$results$ROC)))

p_knn <- ggplot2::ggplot(fit_knn$results, ggplot2::aes(x = k, y = ROC)) +
  ggplot2::geom_line(colour = "purple") +
  ggplot2::geom_point(colour = "purple") +
  ggplot2::geom_vline(xintercept = best_k, linetype = "dashed", colour = "red") +
  ggplot2::annotate("text", x = best_k + 1.5, y = min(fit_knn$results$ROC),
                    label = paste0("K=", best_k), colour = "red", hjust = 0) +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Seleção do K no KNN (10-fold CV)",
                x = "K", y = "AUC-ROC")
print(p_knn)
ggplot2::ggsave("figures/knn_k_selection.png", p_knn, width = 7, height = 5, dpi = 300)

# ============================================================
# 18. AVALIAÇÃO DOS MODELOS NO CONJUNTO DE TESTE
# ============================================================

avaliar_modelo <- function(modelo, nome) {
  pred    <- predict(modelo, newdata = test_data)
  prob    <- predict(modelo, newdata = test_data, type = "prob")[, "Private"]
  cm      <- caret::confusionMatrix(pred, test_data$bl_private, positive = "Private")
  roc_obj <- pROC::roc(test_data$bl_private, prob,
                       levels = c("Public", "Private"), direction = "<", quiet = TRUE)
  data.frame(
    Modelo            = nome,
    Accuracy          = round(as.numeric(cm$overall["Accuracy"]),          4),
    Balanced_Accuracy = round(as.numeric(cm$byClass["Balanced Accuracy"]), 4),
    Sensitivity       = round(as.numeric(cm$byClass["Sensitivity"]),       4),
    Specificity       = round(as.numeric(cm$byClass["Specificity"]),       4),
    F1                = round(as.numeric(cm$byClass["F1"]),                4),
    AUC               = round(as.numeric(pROC::auc(roc_obj)),              4)
  )
}

resultados <- dplyr::bind_rows(
  avaliar_modelo(fit_lda,   "LDA"),
  avaliar_modelo(fit_qda,   "QDA"),
  avaliar_modelo(fit_logit, "Regressão Logística"),
  avaliar_modelo(fit_knn,   paste0("KNN (K=", best_k, ")"))
) |> dplyr::arrange(desc(AUC))

cat("\n=== COMPARAÇÃO DOS MODELOS (conjunto de teste — 30%) ===\n")
print(resultados)
readr::write_csv(resultados, "outputs/model_comparison.csv")

cat(sprintf("\n→ Melhor modelo por AUC : %s (%.4f)\n",
            resultados$Modelo[1], resultados$AUC[1]))
cat(sprintf("→ Melhor modelo por F1  : %s (%.4f)\n",
            resultados$Modelo[which.max(resultados$F1)],
            max(resultados$F1)))

# ============================================================
# 19. MATRIZES DE CONFUSÃO
# ============================================================

for (info in list(
  list(fit_lda,   "LDA"),
  list(fit_qda,   "QDA"),
  list(fit_logit, "Regressão Logística"),
  list(fit_knn,   paste0("KNN (K=", best_k, ")"))
)) {
  cat(sprintf("\n--- Matriz de confusão: %s ---\n", info[[2]]))
  cm_out <- caret::confusionMatrix(predict(info[[1]], test_data),
                                   test_data$bl_private, positive = "Private")
  print(cm_out)
  capture.output(cm_out,
                 file = paste0("outputs/cm_", gsub(" |\\(|\\)", "_", info[[2]]), ".txt"))
}

# ============================================================
# 20. IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================

cat("\n--- Importância das variáveis ---\n")

# LDA — importância via coeficientes LD1 já calculados (secção 14)
p_imp_lda <- ggplot2::ggplot(head(ld1_df, 15),
                             ggplot2::aes(x = reorder(variavel, abs(coef_LD1)),
                                          y = abs(coef_LD1))) +
  ggplot2::geom_col(fill = "#185FA5") +
  ggplot2::coord_flip() +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "LDA — importância (|coef LD1|)",
                x = NULL, y = "|Coeficiente LD1 padronizado|")
print(p_imp_lda)
ggplot2::ggsave("figures/varImp_lda.png", p_imp_lda, width = 8, height = 6, dpi = 300)

# LR — importância via |z-statistic| (evita o erro regMod$residuals do caret)
logit_imp <- broom::tidy(fit_logit$finalModel) |>
  dplyr::filter(term != "(Intercept)") |>
  dplyr::mutate(importance = abs(statistic)) |>
  dplyr::arrange(desc(importance))

p_imp_logit <- ggplot2::ggplot(head(logit_imp, 15),
                               ggplot2::aes(x = reorder(term, importance),
                                            y = importance)) +
  ggplot2::geom_col(fill = "#3B6D11") +
  ggplot2::coord_flip() +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Regressão Logística — importância (|z-statistic|)",
                x = NULL, y = "|z|")
print(p_imp_logit)
ggplot2::ggsave("figures/varImp_logit.png", p_imp_logit, width = 8, height = 6, dpi = 300)

# KNN — importância via AUC univariado por variável (evita bug do caret com twoClassSummary)
knn_imp <- purrr::map_dfr(names(num_model_data), function(v) {
  roc_v <- pROC::roc(test_data$bl_private, test_data[[v]],
                     levels = c("Public", "Private"), direction = "<", quiet = TRUE)
  dplyr::tibble(variavel = v, AUC = as.numeric(pROC::auc(roc_v)))
}) |>
  dplyr::mutate(importance = abs(AUC - 0.5) * 200) |>
  dplyr::arrange(desc(importance))

p_imp_knn <- ggplot2::ggplot(head(knn_imp, 15),
                             ggplot2::aes(x = reorder(variavel, importance),
                                          y = importance)) +
  ggplot2::geom_col(fill = "#7F77DD") +
  ggplot2::coord_flip() +
  ggplot2::theme_bw() +
  ggplot2::labs(title = paste0("KNN (K=", best_k, ") — importância (AUC univariado)"),
                x = NULL, y = "Importância (escala 0-100)")
print(p_imp_knn)
ggplot2::ggsave("figures/varImp_knn.png", p_imp_knn, width = 8, height = 6, dpi = 300)

# ============================================================
# 21. CURVAS ROC SOBREPOSTAS
# ============================================================

roc_lda   <- pROC::roc(test_data$bl_private,
                       predict(fit_lda,   test_data, type="prob")[,"Private"],
                       levels=c("Public","Private"), direction="<", quiet=TRUE)
roc_qda   <- pROC::roc(test_data$bl_private,
                       predict(fit_qda,   test_data, type="prob")[,"Private"],
                       levels=c("Public","Private"), direction="<", quiet=TRUE)
roc_logit <- pROC::roc(test_data$bl_private,
                       predict(fit_logit, test_data, type="prob")[,"Private"],
                       levels=c("Public","Private"), direction="<", quiet=TRUE)
roc_knn   <- pROC::roc(test_data$bl_private,
                       predict(fit_knn,   test_data, type="prob")[,"Private"],
                       levels=c("Public","Private"), direction="<", quiet=TRUE)

png("figures/roc_modelos.png", width = 1200, height = 1000, res = 160)
plot(roc_lda,   col = "#E41A1C", lwd = 2, main = "Curvas ROC — conjunto de teste")
plot(roc_qda,   col = "#377EB8", lwd = 2, add = TRUE)
plot(roc_logit, col = "#4DAF4A", lwd = 2, add = TRUE)
plot(roc_knn,   col = "#984EA3", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         sprintf("LDA   AUC = %.3f", pROC::auc(roc_lda)),
         sprintf("QDA   AUC = %.3f", pROC::auc(roc_qda)),
         sprintf("LR    AUC = %.3f", pROC::auc(roc_logit)),
         sprintf("KNN   AUC = %.3f", pROC::auc(roc_knn))
       ),
       col = c("#E41A1C","#377EB8","#4DAF4A","#984EA3"),
       lwd = 2, bty = "n", cex = 0.9)
dev.off()
cat("\nCurvas ROC guardadas em figures/roc_modelos.png\n")


