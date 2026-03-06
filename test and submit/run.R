# Cntrl + Shift + F10
devtools::document()
# unlink("C:/Users/chedgala/AppData/Local/R/win-library/4.5/00LOCK-ModalCens",
#        recursive = TRUE)
devtools::install(force = TRUE)
library(ModalCens)
example("modal_cens")

