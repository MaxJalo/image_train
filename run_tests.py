# (run_tests)
#!/usr/bin/env python
"""
Скрипт для запуска тестов с различными опциями
Usage: python run_tests.py [options]
"""

import argparse
import subprocess
import sys


def run_command(cmd):
    """Запустить команду и вернуть код выхода"""
    print(f"Запуск: {' '.join(cmd)}")
    print("-" * 80)
    result = subprocess.run(cmd)
    print("-" * 80)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Запуск тестов для проекта image_train")

    parser.add_argument(
        "--all", "-a", action="store_true", help="Запустить все тесты (unit + integration)"
    )

    parser.add_argument("--unit", "-u", action="store_true", help="Запустить только unit тесты")

    parser.add_argument(
        "--integration", "-i", action="store_true", help="Запустить только интеграционные тесты"
    )

    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Запустить тесты с отчётом кода покрытия"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")

    parser.add_argument(
        "--exitfirst", "-x", action="store_true", help="Остановиться на первой ошибке"
    )

    parser.add_argument("--lf", action="store_true", help="Запустить последние неудачные тесты")

    parser.add_argument("--pdb", action="store_true", help="Запустить отладчик при ошибке")

    parser.add_argument("--markers", "-m", type=str, help="Запустить тесты с определённым маркером")

    parser.add_argument("--keyword", "-k", type=str, help="Запустить тесты на основе выражения")

    parser.add_argument("--file", "-f", type=str, help="Запустить тесты из конкретного файла")

    args = parser.parse_args()

    # Построить команду
    cmd = ["pytest"]

    # Определить какие тесты запустить
    if args.file:
        cmd.append(f"tests/{args.file}")
    elif args.integration:
        cmd.append("tests/integration")
    elif args.unit:
        cmd.append("tests/unit")
    else:
        # По умолчанию - все
        cmd.append("tests")

    # Добавить опции
    if args.verbose:
        cmd.append("-v")

    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term-missing"])

    if args.exitfirst:
        cmd.append("-x")

    if args.lf:
        cmd.append("--lf")

    if args.pdb:
        cmd.append("--pdb")

    if args.markers:
        cmd.extend(["-m", args.markers])

    if args.keyword:
        cmd.extend(["-k", args.keyword])

    # Запустить pytest
    exit_code = run_command(cmd)

    if args.coverage:
        print("\n✅ Отчёт о покрытии сохранён в htmlcov/index.html")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
