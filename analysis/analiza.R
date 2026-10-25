# ============================================================
# ANALIZA CNN ARHITEKTURA: CIFAR-100, CIFAR-10 I SVHN
# Excel datoteke i ovu skriptu drzite u istom folderu.
# RStudio: otvorite skriptu i odaberite Source.
# Konzola: source("analiza.R")
# Pakete instalirajte jednom, ako nedostaju:
# install.packages(c("readxl", "lmtest", "sandwich"))
#
# Konacni model je unaprijed odabran. Holdout se ne koristi za
# izbor modela, transformacija, centra ili uklanjanje opazanja.
# CV je NOVA provjera s dokumentiranim seedom, a ne reprodukcija
# nepoznatih izvornih foldova ili cijele povijesti odabira modela.
# ============================================================

# Ostavite NULL za folder skripte. Po potrebi navedite putanju:
# MAPA_PODATAKA <- "C:/.../Dodatak za rad"
MAPA_PODATAKA <- getOption("cnn.mapa", NULL)
POKRENI_CV <- getOption("cnn.cv", TRUE)
CV_SEED <- 20261017L  # Novi seed, nije izvorni seed razvoja modela.
CV_PONAVLJANJA <- 20L
CV_FOLDOVI <- 5L

folder_skripte <- function() {
  frames <- sys.frames()
  paths <- lapply(frames, function(f) f$ofile)
  paths <- Filter(function(x) is.character(x) && length(x) == 1L, paths)
  if (length(paths)) {
    return(dirname(normalizePath(tail(paths, 1L)[[1L]], mustWork = TRUE)))
  }
  arg <- grep("^--file=", commandArgs(), value = TRUE)
  if (length(arg)) {
    return(dirname(normalizePath(sub("^--file=", "", arg[1L]), mustWork = TRUE)))
  }
  getwd()
}

FORMULA_KONACNA <- mean_test_acc ~ log_degree_c + I(log_degree_c^2) +
  normalized_degree_entropy + assortativity_coefficient

provjeri <- function(uvjet, poruka) {
  if (!isTRUE(uvjet)) stop(poruka, call. = FALSE)
}

ucitaj <- function(putanja, sheet, n) {
  provjeri(file.exists(putanja), paste("Nedostaje:", putanja))
  provjeri(sheet %in% readxl::excel_sheets(putanja),
           paste("Nedostaje list", sheet, "u", basename(putanja)))
  x <- as.data.frame(readxl::read_excel(putanja, sheet = sheet))
  kolone <- c("architecture_id", "mean_test_acc", "avg_degree",
              "normalized_degree_entropy", "assortativity_coefficient")
  provjeri(all(kolone %in% names(x)), paste("Nedostaju stupci u", basename(putanja)))
  provjeri(nrow(x) == n, paste("Pogresan broj redaka u", basename(putanja)))
  x$architecture_id <- as.character(x$architecture_id)
  provjeri(!anyNA(x$architecture_id) && all(nzchar(trimws(x$architecture_id))) &&
             !anyDuplicated(x$architecture_id), paste("Neispravni ID-jevi u", basename(putanja)))
  for (v in kolone[-1L]) {
    provjeri(is.numeric(x[[v]]) && all(is.finite(x[[v]])),
             paste("Nenumericne, nedostajuce ili beskonacne vrijednosti:", v, basename(putanja)))
  }
  provjeri(all(x$avg_degree > 0), "Prosjecni stupanj mora biti pozitivan.")
  provjeri(all(x$mean_test_acc >= 0 & x$mean_test_acc <= 1), "Tocnost mora biti u [0,1].")
  provjeri(all(x$normalized_degree_entropy >= 0 & x$normalized_degree_entropy <= 1),
           "Normalizirana entropija mora biti u [0,1].")
  provjeri(all(abs(x$assortativity_coefficient) <= 1), "Asortativnost mora biti u [-1,1].")
  if ("broj_seedova" %in% names(x)) {
    provjeri(!anyNA(x$broj_seedova) && all(x$broj_seedova == 5),
             paste("Nisu svi rezultati prosjek pet seedova:", basename(putanja)))
  }
  x <- x[order(x$architecture_id), , drop = FALSE]
  rownames(x) <- NULL
  x
}

transformiraj <- function(x, centar) {
  x$log_degree <- log10(x$avg_degree)
  x$log_degree_c <- x$log_degree - centar
  x
}

procijeni <- function(formula, x) {
  model <- lm(formula, data = x, na.action = na.fail)
  provjeri(model$rank == length(coef(model)) && all(is.finite(coef(model))),
           "Model ima linearno zavisne prediktore ili neispravne koeficijente.")
  model
}

metrike <- function(y, pred, trening_prosjek = NULL) {
  provjeri(length(y) == length(pred) && all(is.finite(y)) && all(is.finite(pred)),
           "Neispravne stvarne ili predvidjene vrijednosti.")
  sse <- sum((y - pred)^2)
  sst <- sum((y - mean(y))^2)
  provjeri(sst > 0, "R2 nije definiran za konstantnu ovisnu varijablu.")
  sigurna_korelacija <- function(method) {
    if (sd(pred) == 0) return(NA_real_)
    cor(y, pred, method = method)
  }
  r <- data.frame(RMSE = sqrt(mean((y - pred)^2)), MAE = mean(abs(y - pred)),
                  R2 = 1 - sse / sst, Spearman = sigurna_korelacija("spearman"),
                  Pearson = sigurna_korelacija("pearson"))
  r$RMSE_postotni_bodovi <- 100 * r$RMSE
  r$MAE_postotni_bodovi <- 100 * r$MAE
  if (!is.null(trening_prosjek)) {
    nazivnik <- sum((y - trening_prosjek)^2)
    r$Q2_external <- if (nazivnik > 0) 1 - sse / nazivnik else NA_real_
  }
  r
}

koeficijenti_hc3 <- function(model) {
  V <- sandwich::vcovHC(model, type = "HC3")
  ct <- lmtest::coeftest(model, vcov. = V, df = df.residual(model))
  kriticna_t <- qt(0.975, df.residual(model))
  data.frame(prediktor = rownames(ct), procjena = ct[, 1], HC3_SE = ct[, 2],
             donja_95_CI = ct[, 1] - kriticna_t * ct[, 2],
             gornja_95_CI = ct[, 1] + kriticna_t * ct[, 2],
             t = ct[, 3], p = ct[, 4], row.names = NULL)
}

vif_tol <- function(model) {
  X <- as.data.frame(model.matrix(model)[, -1L, drop = FALSE])
  imena <- names(X)
  names(X) <- paste0("v", seq_along(X))
  do.call(rbind, lapply(seq_along(X), function(i) {
    pomocni <- lm(X[[i]] ~ ., data = X[, -i, drop = FALSE])
    R2 <- summary(pomocni)$r.squared
    data.frame(prediktor = imena[i], R2 = R2, TOL = 1 - R2, VIF = 1 / (1 - R2))
  }))
}

deskriptivna <- function(x) {
  vars <- c("mean_test_acc", "avg_degree", "log_degree",
            "normalized_degree_entropy", "assortativity_coefficient")
  do.call(rbind, lapply(vars, function(v) {
    z <- x[[v]]
    data.frame(varijabla = v, N = length(z), prosjek = mean(z), SD = sd(z),
               medijan = median(z), minimum = min(z), maksimum = max(z))
  }))
}

spremi_csv <- function(x, datoteka) {
  write.csv2(x, datoteka, row.names = FALSE, fileEncoding = "UTF-8", na = "NA")
}

grafovi <- function(model, holdout, pred, naziv, datoteka) {
  pdf(datoteka, width = 10, height = 8, onefile = TRUE)
  on.exit(dev.off(), add = TRUE)
  par(mfrow = c(2, 2), oma = c(0, 0, 2, 0))
  plot(model, which = c(1, 2, 3, 5), ask = FALSE)
  mtext(paste(naziv, "- razvojni skup"), outer = TRUE)
  par(mfrow = c(1, 1), oma = c(0, 0, 0, 0))
  lim <- range(100 * c(holdout$mean_test_acc, pred))
  plot(100 * holdout$mean_test_acc, 100 * pred, pch = 19, col = "#246B83",
       xlim = lim, ylim = lim, asp = 1, xlab = "Stvarna tocnost (%)",
       ylab = "Predvidjena tocnost (%)", main = paste(naziv, "- holdout, N = 54"))
  abline(0, 1, col = "#B64E38", lty = 2, lwd = 2)
}

# NOVA stratificirana CV: po vrijednosti y sortirane grupe od pet
# arhitektura rasporedjuju se nasumicno po jednom u svaki fold.
# Foldovi su isti za sve usporedjene formule unutar skupa/ponavljanja.
# Holdout nije argument funkcije i ne sudjeluje u CV-u.
ponovna_cv <- function(dev, formule, seed, ponavljanja, k) {
  stari_rng <- RNGkind()
  imao_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (imao_seed) stari_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    do.call(RNGkind, as.list(stari_rng))
    if (imao_seed) assign(".Random.seed", stari_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
      rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
  red <- order(dev$mean_test_acc, dev$architecture_id)
  blokovi <- split(red, ceiling(seq_along(red) / k))
  svi_foldovi <- svi_centri <- sve_metrike <- sve_predikcije <- list()
  for (rep in seq_len(ponavljanja)) {
    fold <- integer(nrow(dev))
    for (blok in blokovi) fold[blok] <- sample.int(k, length(blok))
    provjeri(diff(range(tabulate(fold, nbins = k))) <= 1, "Neuravnotezeni CV foldovi.")
    svi_foldovi[[rep]] <- data.frame(ponavljanje = rep, architecture_id = dev$architecture_id,
                                     fold = fold)
    pred <- matrix(NA_real_, nrow(dev), length(formule), dimnames = list(NULL, names(formule)))
    for (f in seq_len(k)) {
      ucenje <- dev[fold != f, , drop = FALSE]
      validacija <- dev[fold == f, , drop = FALSE]
      centar <- mean(log10(ucenje$avg_degree))
      ucenje <- transformiraj(ucenje, centar)
      validacija <- transformiraj(validacija, centar)
      svi_centri[[length(svi_centri) + 1L]] <- data.frame(
        ponavljanje = rep, fold = f, N_ucenje = nrow(ucenje),
        N_validacija = nrow(validacija), centar_log = centar)
      for (m in names(formule)) {
        fit <- procijeni(formule[[m]], ucenje)
        pred[fold == f, m] <- predict(fit, newdata = validacija)
      }
    }
    for (m in names(formule)) {
      sve_metrike[[length(sve_metrike) + 1L]] <- cbind(
        data.frame(model = m, ponavljanje = rep), metrike(dev$mean_test_acc, pred[, m]))
      sve_predikcije[[length(sve_predikcije) + 1L]] <- data.frame(
        model = m, ponavljanje = rep, architecture_id = dev$architecture_id,
        fold = fold, stvarna_tocnost = dev$mean_test_acc, predvidjena_tocnost = pred[, m])
    }
  }
  values <- do.call(rbind, sve_metrike)
  pokazatelji <- setdiff(names(values), c("model", "ponavljanje"))
  sazetak <- do.call(rbind, lapply(names(formule), function(m) {
    z <- values[values$model == m, , drop = FALSE]
    do.call(rbind, lapply(pokazatelji, function(v) {
      data.frame(model = m, pokazatelj = v, prosjek = mean(z[[v]]), SD = sd(z[[v]]))
    }))
  }))
  list(metrike = values, sazetak = sazetak, foldovi = do.call(rbind, svi_foldovi),
       centri = do.call(rbind, svi_centri), predikcije = do.call(rbind, sve_predikcije))
}

analiziraj_sve <- function(mapa, pokreni_cv = TRUE) {
  paketi <- c("readxl", "lmtest", "sandwich")
  nedostaju <- paketi[!vapply(paketi, requireNamespace, logical(1), quietly = TRUE)]
  provjeri(!length(nedostaju), paste("Instalirajte pakete:", paste(nedostaju, collapse = ", ")))
  mapa <- normalizePath(mapa, winslash = "/", mustWork = TRUE)
  konfiguracija <- data.frame(
    kljuc = c("cifar100", "cifar10", "svhn"), skup = c("CIFAR-100", "CIFAR-10", "SVHN"),
    dev = c("cifar100_development162_modularity5.xlsx", "cifar10_development162.xlsx", "svhn_development162.xlsx"),
    holdout = c("cifar100_final_holdout54.xlsx", "cifar10_final_holdout54.xlsx", "svhn_final_holdout54.xlsx"),
    dev_sheet = c("development_162", "Sheet1", "development_162"),
    holdout_sheet = c("final_holdout_54", "Sheet1", "final_holdout_54"))
  podaci <- list()
  for (i in seq_len(nrow(konfiguracija))) {
    cfg <- konfiguracija[i, ]
    dev <- ucitaj(file.path(mapa, cfg$dev), cfg$dev_sheet, 162L)
    holdout <- ucitaj(file.path(mapa, cfg$holdout), cfg$holdout_sheet, 54L)
    provjeri(!length(intersect(dev$architecture_id, holdout$architecture_id)),
             paste("Preklapanje development/holdout:", cfg$skup))
    podaci[[cfg$kljuc]] <- list(dev = dev, holdout = holdout)
  }
  topologija <- c("avg_degree", "normalized_degree_entropy", "assortativity_coefficient")
  for (kljuc in c("cifar10", "svhn")) for (dio in c("dev", "holdout")) {
    a <- podaci$cifar100[[dio]]
    b <- podaci[[kljuc]][[dio]]
    provjeri(identical(a$architecture_id, b$architecture_id),
             paste("Razliciti identifikatori arhitektura:", kljuc, dio))
    for (v in topologija) provjeri(isTRUE(all.equal(a[[v]], b[[v]], tolerance = 1e-9)),
                                  paste("Topoloski prediktori nisu jednaki:", kljuc, dio, v))
  }

  # Novo poddirektorij za svako pokretanje: bez prepisivanja starih rezultata.
  korijen <- file.path(mapa, "rezultati")
  dir.create(korijen, showWarnings = FALSE)
  izlaz <- tempfile(paste0("analiza_", format(Sys.time(), "%Y%m%d_%H%M%S"), "_"), tmpdir = korijen)
  provjeri(dir.create(izlaz), "Nije moguce stvoriti folder rezultata.")
  files <- file.path(mapa, c(konfiguracija$dev, konfiguracija$holdout))
  spremi_csv(data.frame(datoteka = basename(files), md5 = unname(tools::md5sum(files))),
             file.path(izlaz, "ulazne_datoteke_md5.csv"))
  spremi_csv(konfiguracija, file.path(izlaz, "ulazni_listovi.csv"))
  writeLines(c(
    "Konacna specifikacija je fiksirana prema dostavljenom izvornom R kodu.",
    "Nema automatskog izbora modela, izbacivanja opazanja ili podesavanja na holdoutu.",
    "R2 holdout = 1 - SSE/sum((y_holdout - mean(y_holdout))^2).",
    "Q2 external = 1 - SSE/sum((y_holdout - mean(y_development))^2).",
    "Tocnost, RMSE i MAE su u [0,1] skali; stupci postotni_bodovi mnozeni su sa 100.",
    "HC3 intervali i testovi koriste t raspodjelu s n-p stupnjeva slobode.",
    "VIF/TOL odnose se na cetiri stupca modela, ukljucujuci kvadratni clan.",
    "Pragovi utjecaja su orijentacijski; nijedna arhitektura se automatski ne uklanja.",
    paste("Nova CV provjera ukljucena:", pokreni_cv),
    paste("Novi CV seed:", CV_SEED, "; dodatak 0/1/2 za CIFAR-100/CIFAR-10/SVHN."),
    "CV: 20 ponavljanja x 5 foldova, sortirani blokovi po pet prema y, nasumicna dodjela.",
    "Pokazatelji CV-a racunaju se na svih 162 OOF predikcija po ponavljanju, zatim mean/SD.",
    "SD kroz ponavljanja nije standardna pogreska ni interval pouzdanosti.",
    "CV je nova provjera; izvorni foldovi/seedovi nisu dostavljeni.",
    "Dodatne CV usporedbe na CIFAR-100 nisu rekonstrukcija povijesti odabira kandidata.",
    "Nisu rekonstruirani izvorni kandidatski modeli s parametrima i modularnoscu.",
    "Vremena preslikavanja/treniranja nisu analizirana u ovoj skripti."
  ), file.path(izlaz, "PROCITAJ.txt"))

  rezultati <- list()
  svi_holdout <- svi_fit <- svi_hc3 <- list()
  for (i in seq_len(nrow(konfiguracija))) {
    cfg <- konfiguracija[i, ]
    cat("\nAnaliza:", cfg$skup, "\n")
    d <- file.path(izlaz, cfg$kljuc)
    dir.create(d)
    dev <- podaci[[cfg$kljuc]]$dev
    holdout <- podaci[[cfg$kljuc]]$holdout
    centar <- mean(log10(dev$avg_degree))
    dev <- transformiraj(dev, centar)
    holdout <- transformiraj(holdout, centar)
    model <- procijeni(FORMULA_KONACNA, dev)
    hc3 <- koeficijenti_hc3(model)
    pred <- as.numeric(predict(model, newdata = holdout))
    met <- cbind(data.frame(skup = cfg$skup, N = nrow(holdout), centar_log = centar),
                 metrike(holdout$mean_test_acc, pred, mean(dev$mean_test_acc)))
    names(met)[names(met) == "R2"] <- "R2_holdout"
    sm <- summary(model)
    fit <- data.frame(skup = cfg$skup, N = nobs(model), centar_log = centar,
                      R2 = sm$r.squared, prilagodjeni_R2 = sm$adj.r.squared,
                      rezidualna_standardna_pogreska = sm$sigma,
                      rezidualna_pogreska_postotni_bodovi = 100 * sm$sigma)
    predikcije <- data.frame(architecture_id = holdout$architecture_id,
                             stvarna_tocnost = holdout$mean_test_acc,
                             predvidjena_tocnost = pred, rezidual = holdout$mean_test_acc - pred)
    bp <- lmtest::bptest(model)  # Zadani studentizirani (Koenker) oblik.
    sw <- shapiro.test(residuals(model))
    n <- nobs(model)
    p <- length(coef(model))
    utjecaj <- data.frame(architecture_id = dev$architecture_id,
                          rezidual = residuals(model), studentizirani = rstudent(model),
                          Cook_D = cooks.distance(model), leverage = hatvalues(model),
                          Cook_granica = 4 / n, leverage_granica = 2 * p / n)
    utjecaj$flag_studentizirani <- abs(utjecaj$studentizirani) > 3
    utjecaj$flag_Cook <- utjecaj$Cook_D > 4 / n
    utjecaj$flag_leverage <- utjecaj$leverage > 2 * p / n
    dijag <- data.frame(skup = cfg$skup, BP = unname(bp$statistic), BP_df = unname(bp$parameter),
                        BP_p = bp$p.value, Shapiro_W = unname(sw$statistic), Shapiro_p = sw$p.value,
                        broj_studentizirani_iznad3 = sum(utjecaj$flag_studentizirani),
                        broj_Cook_iznad_granice = sum(utjecaj$flag_Cook),
                        broj_leverage_iznad_granice = sum(utjecaj$flag_leverage),
                        najveci_Cook_D = max(utjecaj$Cook_D))
    desc <- deskriptivna(dev)
    vars <- c("mean_test_acc", "avg_degree", "log_degree",
              "normalized_degree_entropy", "assortativity_coefficient")
    korelacije <- cor(dev[, vars], method = "spearman")
    spremi_csv(data.frame(varijabla = rownames(korelacije), korelacije, check.names = FALSE),
               file.path(d, "Spearman_razvojni.csv"))
    spremi_csv(desc, file.path(d, "deskriptivna_razvojni.csv"))
    spremi_csv(hc3, file.path(d, "koeficijenti_HC3.csv"))
    spremi_csv(fit, file.path(d, "prilagodba_razvojni.csv"))
    spremi_csv(vif_tol(model), file.path(d, "VIF_TOL.csv"))
    spremi_csv(dijag, file.path(d, "dijagnostika_razvojni.csv"))
    spremi_csv(utjecaj, file.path(d, "utjecajna_opazanja_sve_arhitekture.csv"))
    spremi_csv(predikcije, file.path(d, "holdout_predikcije.csv"))
    spremi_csv(met, file.path(d, "holdout_metrike.csv"))
    writeLines(capture.output(summary(model)), file.path(d, "lm_summary.txt"))
    grafovi(model, holdout, pred, cfg$skup, file.path(d, "dijagnostika_i_holdout.pdf"))
    # Paket cuva i centar, nuzen za buduce predikcije iz avg_degree.
    saveRDS(list(model = model, centar_log = centar, formula = FORMULA_KONACNA,
                 development_ids = dev$architecture_id, holdout_ids = holdout$architecture_id),
            file.path(d, "model_i_transformacija.rds"))

    cv <- NULL
    if (pokreni_cv) {
      formule <- list(konacni = FORMULA_KONACNA)
      if (cfg$kljuc == "cifar100") {
        provjeri(is.numeric(dev$degree_entropy) && all(is.finite(dev$degree_entropy)),
                 "Za usporedbu entropija nedostaje degree_entropy.")
        formule$bez_kvadratnog <- mean_test_acc ~ log_degree_c +
          normalized_degree_entropy + assortativity_coefficient
        formule$nenormalizirana_entropija <- mean_test_acc ~ log_degree_c +
          I(log_degree_c^2) + degree_entropy + assortativity_coefficient
      }
      cat("Nova CV provjera 20 x 5:", cfg$skup, "\n")
      cv <- ponovna_cv(dev, formule, CV_SEED + i - 1L, CV_PONAVLJANJA, CV_FOLDOVI)
      cv_dir <- file.path(d, "CV_NOVA_PROVJERA")
      dir.create(cv_dir)
      for (naziv in names(cv)) spremi_csv(cv[[naziv]], file.path(cv_dir, paste0(naziv, ".csv")))
      saveRDS(list(formule = formule, seed = CV_SEED + i - 1L,
                   ponavljanja = CV_PONAVLJANJA, k = CV_FOLDOVI), file.path(cv_dir, "postavke.rds"))
    }
    svi_holdout[[i]] <- met
    svi_fit[[i]] <- fit
    svi_hc3[[i]] <- cbind(skup = cfg$skup, hc3)
    rezultati[[cfg$kljuc]] <- list(model = model, centar_log = centar, razvojni = dev,
                                   holdout = holdout, metrike = met, koeficijenti = hc3,
                                   deskriptivna = desc, predikcije = predikcije, CV = cv)
    print(met, row.names = FALSE)
  }

  svi <- do.call(rbind, svi_holdout)
  spremi_csv(svi, file.path(izlaz, "holdout_metrike_svi_skupovi.csv"))
  spremi_csv(do.call(rbind, svi_fit), file.path(izlaz, "prilagodba_svi_skupovi.csv"))
  spremi_csv(do.call(rbind, svi_hc3), file.path(izlaz, "koeficijenti_HC3_svi_skupovi.csv"))
  fmt <- function(x, d = 4) formatC(x, format = "f", digits = d, decimal.mark = ",", big.mark = ".")
  desc <- rezultati$cifar100$deskriptivna
  dec <- c(4, 2, 4, 4, 4)
  nazivi <- c("Prosjecna testna tocnost", "Prosjecni stupanj", "Log10 prosjecnog stupnja",
              "Normalizirana entropija stupnja", "Asortativnost prema stupnju")
  tab1 <- data.frame(Varijabla = nazivi,
                     `Prosjek +/- SD` = mapply(function(m, s, d) paste(fmt(m, d), "+/-", fmt(s, d)),
                                              desc$prosjek, desc$SD, dec),
                     Medijan = mapply(fmt, desc$medijan, dec),
                     Raspon = mapply(function(lo, hi, d) paste(fmt(lo, d), fmt(hi, d), sep = " do "),
                                      desc$minimum, desc$maksimum, dec), check.names = FALSE)
  hc <- rezultati$cifar100$koeficijenti
  tab2 <- data.frame(Prediktor = c("Konstanta", "Centrirani log10 prosjecnog stupnja",
                                  "Kvadratni clan", nazivi[4:5]),
                     Procjena = fmt(hc$procjena), `HC3 SE` = fmt(hc$HC3_SE),
                     `95 % CI` = paste0("[", fmt(hc$donja_95_CI), "; ", fmt(hc$gornja_95_CI), "]"),
                     p = ifelse(hc$p < 0.001, "< 0,001", fmt(hc$p, 3)), check.names = FALSE)
  tab3 <- data.frame(Skup = svi$skup, R2_holdout = fmt(svi$R2_holdout, 3),
                     `Spearman rho` = fmt(svi$Spearman, 3),
                     `RMSE (p. b.)` = fmt(svi$RMSE_postotni_bodovi, 3), check.names = FALSE)
  spremi_csv(tab1, file.path(izlaz, "Table_S1_descriptive_CIFAR100.csv"))
  spremi_csv(tab2, file.path(izlaz, "Tablica_2_koeficijenti_CIFAR100.csv"))
  spremi_csv(tab3, file.path(izlaz, "Tablica_3_holdout_svi_skupovi.csv"))
  writeLines(capture.output(sessionInfo()), file.path(izlaz, "sessionInfo.txt"))
  cat("\nTABLICA S1\n"); print(tab1, row.names = FALSE)
  cat("\nTABLICA 2\n"); print(tab2, row.names = FALSE)
  cat("\nTABLICA 3\n"); print(tab3, row.names = FALSE)
  cat("\nRezultati su spremljeni u:\n", izlaz, "\n")
  rezultati$izlazna_mapa <- izlaz
  rezultati
}

if (is.null(MAPA_PODATAKA)) MAPA_PODATAKA <- folder_skripte()
rezultati <- analiziraj_sve(MAPA_PODATAKA, POKRENI_CV)
