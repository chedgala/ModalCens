# 1. Revisión local estricta (simula el entorno de CRAN)
devtools::check(cran = TRUE)

# 2. Pruebas en servidores externos de CRAN (Windows y Mac)
# Te enviarán un correo con los resultados en unos minutos.
devtools::check_win_devel()
devtools::check_mac_release()

#3. Crear el archivo de comentarios para los mantenedores de CRAN
# Esto abrirá un archivo de texto. Debes llenarlo.
usethis::use_cran_comments()

# 4. Enviar oficialmente a CRAN
# Este comando correrá verificaciones finales y te pedirá confirmación interactiva.
devtools::release()
