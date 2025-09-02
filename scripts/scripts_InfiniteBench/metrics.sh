
results_dir=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --results_dir) results_dir="$2"; shift 2;; 
    --new_method) new_method="$2"; shift 2;;
    --switch) switch="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

python3 eval_infinitebench.py \
    --results_dir ${results_dir} \
    --new_method ${new_method} \
    --switch ${switch}
