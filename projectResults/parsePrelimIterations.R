library(tidyverse)
dir <- file.path("C:", "Users", "Ben_Sepanski",
                 "OneDrive", "Math_Files", "Homework",
                 "Compilers-20F", "ProjectReport")
result_files_dir <- file.path(dir, "PrelimResultFiles")
result_text <- result_files_dir %>% 
               dir() %>%
               str_c(result_files_dir, '/', .) %>%
               sapply(readr::read_file)

devices <- c(rep("Tesla K80", 2), rep("GeForce GTX", 2), rep("Tesla K80", 2))

results <- tibble::tibble(device = str_extract_all(result_text, "(?<=Device )\\d"),
                          time_in_ms = str_extract_all(result_text, "(?<=Total time: )\\d+(?= ms)"),
                          time_in_ns = str_extract_all(result_text, "(?<=Total time: )\\d+(?= ns)"),
                          method = str_extract_all(result_text, "(SYCL Data-Driven|SYCL Topology-Driven|Lonestar)")
                          ) %>%
           map(~.x %>% flatten() %>% unlist()) %>%
           tibble::as_tibble() %>%
           dplyr::mutate(time_in_ms = time_in_ms %>% as.numeric(),
                         time_in_ns = time_in_ns %>% as.numeric(),
                         device = devices[as.numeric(device) + 1],
                         graph = result_text %>%
                                  str_extract("(?<=INPUTGRAPH ).*(?=\\.gr(\n|\r))") %>%
                                  map(~rep(.x, 9)) %>%
                                  unlist()
                          ) %>%
           dplyr::mutate(graph = str_extract(graph, "[^/]*$"))

readr::write_csv(results, file=file.path(dir, "prelimResults.csv"))