#!/usr/bin/env bash

set -euo pipefail

if command -v py >/dev/null 2>&1 && py -3 -V >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python3 >/dev/null 2>&1 && python3 -V >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1 && python -V >/dev/null 2>&1; then
  PYTHON_CMD=(python)
else
  echo "Erro: Python executavel nao encontrado."
  echo "No Windows, instala Python e/ou usa o launcher 'py'."
  exit 1
fi

run_task() {
  local task_number="$1"
  local file="Tarefa ${task_number}.py"

  if [[ ! -f "$file" ]]; then
    echo "Aviso: ficheiro '$file' nao existe, a saltar."
    return
  fi

  echo
  echo "============================================================"
  echo "A executar: $file"
  echo "============================================================"
  "${PYTHON_CMD[@]}" "$file"
}

print_help() {
  echo "Uso: ./run_project.sh [1|2|3|4|5|6|all]"
  echo
  echo "Exemplos:"
  echo "  ./run_project.sh 2"
  echo "  ./run_project.sh 4"
  echo "  ./run_project.sh all"
}

if [[ $# -eq 0 ]]; then
  print_help
  exit 0
fi

case "$1" in
  1|2|3|4|5|6)
    run_task "$1"
    ;;
  all)
    for n in 1 2 3 4 5 6; do
      run_task "$n"
    done
    ;;
  -h|--help|help)
    print_help
    ;;
  *)
    echo "Argumento invalido: '$1'"
    print_help
    exit 1
    ;;
esac
