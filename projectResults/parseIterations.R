library(tidyverse)
dir <- file.path("C:", "Users", "Ben_Sepanski",
                 "OneDrive", "Math_Files", "Homework",
                 "Compilers-20F", "ProjectReport")
result_files_dir <- file.path(dir, "ResultFiles")
result_text <- result_files_dir %>% 
  dir() %>%
  str_c(result_files_dir, '/', .) %>%
  sapply(readr::read_file)

devices <- c(rep("Tesla K80", 2), rep("GeForce GTX", 2), rep("Tesla K80", 2))

# 3 lonestar runs, 3 runs for each data-driven and topology-driven
# implementation with block-sizes of 2, 4, 8, 16, 32
num_iters_per_file <- 3 + 3 * 2 * 5

results <- tibble::tibble(device = str_extract_all(result_text, "(?<=Device )\\d"),
                          time_in_ms = str_extract_all(result_text, "(?<=Total time: )(\\d+|[nN][aA])(?= ms)"),
                          time_in_ns = str_extract_all(result_text, "(?<=Total time: )(\\d+|[nN][aA])(?= ns)"),
                          method = str_extract_all(result_text, "(SYCL Data-Driven|SYCL Topology-Driven|Lonestar)"),
                          num_work_groups = str_extract_all(result_text, "((?<=NUM WORK GROUPS: )\\d+|Lonestar)")
                          ) %>%
  map(~.x %>% flatten() %>% unlist()) %>% 
  tibble::as_tibble() %>%
  dplyr::mutate(time_in_ms = time_in_ms %>% dplyr::na_if("NA") %>% as.numeric(),
                time_in_ns = time_in_ns %>% dplyr::na_if("NA") %>% as.numeric(),
                num_work_groups = num_work_groups %>% dplyr::na_if("Lonestar") %>% as.numeric(),
                device = devices[as.numeric(device) + 1],
                graph = result_text %>%
                  str_extract("(?<=INPUTGRAPH ).*(?=\\.gr(\n|\r))") %>%
                  map(~rep(.x, num_iters_per_file)) %>%
                  unlist(),
                nnodes = result_text %>%
                  str_extract("(?<=nnodes=)\\d+") %>%
                  map(~rep(.x, num_iters_per_file)) %>%
                  unlist() %>%
                  as.numeric(),
                nedges = result_text %>%
                  str_extract("(?<=nedges=)\\d+") %>%
                  map(~rep(.x, num_iters_per_file)) %>%
                  unlist() %>%
                  as.numeric(),
                problem = result_text %>%
                  str_extract("PR") %>%
                  map(~rep(.x, num_iters_per_file)) %>%
                  unlist() %>% 
                  map(~if(is.na(.x)) {return("BFS")} else {return("PR")}) %>%
                  unlist()
  ) %>%
  dplyr::mutate(graph = str_extract(graph, "[^/]*$"))

readr::write_csv(results, file=file.path(dir, "results.csv"))