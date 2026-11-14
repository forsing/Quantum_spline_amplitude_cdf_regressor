# Quantum_spline_amplitude_cdf_regressor



import random
import warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import SplineTransformer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore", category=UserWarning)

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
WINDOW_LEN = 3
KNOTS_GRID = (4, 6)
PENALTY_GRID = (1e-1, 1.0, 5.0)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < WINDOW_LEN + 30:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_spline_inputs(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_total = len(draws)
    x_rows = []
    y_rows = []
    for t in range(WINDOW_LEN, n_total):
        win = draws[t - WINDOW_LEN : t].astype(np.float64) / float(MAX_NUM)
        x_rows.append(win.reshape(-1))
        y_rows.append(draws.__getitem__(t))
    return np.array(x_rows, dtype=np.float64), np.array(y_rows, dtype=np.int64)


def build_controlled_spline_circuit(spline_row: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS)
    segments = np.array_split(spline_row, NUM_QUBITS)

    for q in range(NUM_QUBITS):
        seg = segments.__getitem__(q)
        s_mean = float(np.mean(seg))
        s_max = float(np.max(seg))
        s_energy = float(np.sqrt(np.mean(seg ** 2)))
        qc.ry(np.pi * s_mean, q)
        qc.rz(2.0 * np.pi * s_max, q)
        qc.rx(np.pi * s_energy, q)

    for q in range(NUM_QUBITS - 1):
        ctrl_angle = np.pi * float(np.mean(segments.__getitem__(q)))
        qc.cry(ctrl_angle, q, q + 1)

    ring_angle = np.pi * float(np.mean(segments.__getitem__(NUM_QUBITS - 1)))
    qc.cry(ring_angle, NUM_QUBITS - 1, 0)

    for q in range(0, NUM_QUBITS - 1, 2):
        qc.cz(q, q + 1)
    for q in range(1, NUM_QUBITS - 1, 2):
        qc.cz(q, q + 1)

    return qc


def extract_spline_quantum_amplitudes(spline_matrix: np.ndarray) -> np.ndarray:
    n_samples = int(spline_matrix.shape.__getitem__(0))
    zero_sv = Statevector.from_int(0, dims=STATE_DIM)
    amp_states = np.empty((n_samples, STATE_DIM), dtype=np.complex128)
    for i in range(n_samples):
        qc = build_controlled_spline_circuit(spline_matrix.__getitem__(i))
        amp_states[i] = zero_sv.evolve(qc).data
    return amp_states


def amplitudes_to_spline_basis(amp_states: np.ndarray) -> np.ndarray:
    re = np.real(amp_states)
    im = np.imag(amp_states)
    prob = np.abs(amp_states) ** 2
    bias = np.ones((len(amp_states), 1), dtype=np.float64)
    return np.hstack([bias, re, im, prob])


def build_empirical_cdf_targets(y_draws: np.ndarray) -> np.ndarray:
    n_samples = len(y_draws)
    num_grid = np.arange(1, MAX_NUM + 1, dtype=np.int64)[None, None, :]
    y_exp = y_draws[:, :, None]
    cdf_targets = (y_exp <= num_grid).astype(np.float64)
    return cdf_targets.reshape(n_samples, NUM_POSITIONS * MAX_NUM)


def fit_penalized_spline_cdf_solver(
    basis_train: np.ndarray,
    y_train: np.ndarray,
    penalty: float,
) -> np.ndarray:
    n_features = int(basis_train.shape.__getitem__(1))
    targets = build_empirical_cdf_targets(y_train)
    gram = basis_train.T @ basis_train
    reg_eye = penalty * np.eye(n_features, dtype=np.float64)
    reg_eye.flat.__setitem__(0, 1e-6)
    rhs = basis_train.T @ targets
    weights = np.linalg.solve(gram + reg_eye, rhs)
    return weights


def cdf_to_monotonic_pmf(raw_cdf_flat: np.ndarray) -> np.ndarray:
    n_samples = int(raw_cdf_flat.shape.__getitem__(0))
    cdf = raw_cdf_flat.reshape(n_samples, NUM_POSITIONS, MAX_NUM)
    cdf_clipped = np.clip(cdf, 0.0, 1.0)
    cdf_mono = np.maximum.accumulate(cdf_clipped, axis=2)

    pmf = np.empty_like(cdf_mono)
    pmf[:, :, 0] = cdf_mono[:, :, 0]
    pmf[:, :, 1:] = np.maximum(cdf_mono[:, :, 1:] - cdf_mono[:, :, :-1], 0.0)

    num_grid = np.arange(1, MAX_NUM + 1, dtype=np.int64)[None, :]
    pos_idx = np.arange(NUM_POSITIONS, dtype=np.int64)[:, None]
    min_allowed = pos_idx + 1
    max_allowed = MAX_NUM - (NUM_POSITIONS - 1 - pos_idx)
    pos_mask = (num_grid >= min_allowed) & (num_grid <= max_allowed)

    pmf_masked = np.where(pos_mask[None, :, :], pmf + 1e-4, 0.0)
    denom = np.sum(pmf_masked, axis=2, keepdims=True)
    uniform_fb = pos_mask[None, :, :].astype(np.float64) / np.sum(pos_mask, axis=1, keepdims=True)[None, :, :]
    pmf_norm = np.where(denom > 1e-15, pmf_masked / np.maximum(denom, 1e-15), uniform_fb)
    return pmf_norm


def evaluate_cdf_nll(weights: np.ndarray, basis_eval: np.ndarray, y_eval: np.ndarray) -> float:
    raw_cdf = basis_eval @ weights
    pmf = cdf_to_monotonic_pmf(raw_cdf)
    n_samples = len(y_eval)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_eval - 1
    chosen = pmf[sample_idx, pos_idx, target_idx]
    return -float(np.mean(np.sum(np.log(np.maximum(chosen, 1e-12)), axis=1)))


def inverse_cdf_ordered_decode(raw_cdf_single: np.ndarray) -> list[int]:
    cdf = np.clip(raw_cdf_single.reshape(NUM_POSITIONS, MAX_NUM), 0.0, 1.0)
    cdf_mono = np.maximum.accumulate(cdf, axis=1)

    chosen_nums: list[int] = []
    prev_num = 0
    for pos in range(NUM_POSITIONS):
        remaining = NUM_POSITIONS - 1 - pos
        low = prev_num + 1
        high = MAX_NUM - remaining

        row_cdf = cdf_mono.__getitem__(pos)
        low_cdf = float(row_cdf.__getitem__(low - 2)) if low > 1 else 0.0
        high_cdf = float(row_cdf.__getitem__(high - 1))

        best_num = low
        best_jump = -1.0
        for num in range(low, high + 1):
            prev_c = float(row_cdf.__getitem__(num - 2)) if num > 1 else 0.0
            curr_c = float(row_cdf.__getitem__(num - 1))
            jump = curr_c - prev_c
            if jump > best_jump:
                best_jump = jump
                best_num = num

        if best_jump <= 1e-9:
            target_q = 0.5 * (low_cdf + high_cdf)
            best_dist = float("inf")
            for num in range(low, high + 1):
                dist = abs(float(row_cdf.__getitem__(num - 1)) - target_q)
                if dist < best_dist:
                    best_dist = dist
                    best_num = num

        chosen_nums.append(best_num)
        prev_num = best_num

    if len(set(chosen_nums)) != NUM_POSITIONS:
        raise RuntimeError("Neuspesno inverzno dekodiranje 7 razlicitih brojeva.")
    return chosen_nums


def main() -> None:
    print("Metoda: Qiskit Cubic B-Spline Controlled-Amplitude CDF Regressor (1x6 qubita)")
    print("Posebnost: Hronoloski B-Spline cvorovi + kontrolisano CRY kolo + zatvorena CDF regresija na simulatoru")

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        x_raw, y_all = build_spline_inputs(draws)
        n_samples = int(x_raw.shape.__getitem__(0))

        train_end = int(0.75 * n_samples)
        val_end = int(0.88 * n_samples)

        x_tr, y_tr = x_raw[:train_end], y_all[:train_end]
        x_va, y_va = x_raw[train_end:val_end], y_all[train_end:val_end]
        x_te, y_te = x_raw[val_end:], y_all[val_end:]

        best_knots = int(KNOTS_GRID.__getitem__(0))
        best_penalty = float(PENALTY_GRID.__getitem__(0))
        best_val_nll = float("inf")

        for n_knots in KNOTS_GRID:
            spline_tf = SplineTransformer(
                n_knots=n_knots,
                degree=3,
                knots="quantile",
                include_bias=False,
            )
            spline_tf.fit(x_tr)
            b_tr = amplitudes_to_spline_basis(extract_spline_quantum_amplitudes(spline_tf.transform(x_tr)))
            b_va = amplitudes_to_spline_basis(extract_spline_quantum_amplitudes(spline_tf.transform(x_va)))

            for penalty in PENALTY_GRID:
                weights = fit_penalized_spline_cdf_solver(b_tr, y_tr, penalty=penalty)
                val_nll = evaluate_cdf_nll(weights, b_va, y_va)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_knots = n_knots
                    best_penalty = penalty

        eval_spline = SplineTransformer(
            n_knots=best_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        )
        eval_spline.fit(x_tr)
        b_tr_best = amplitudes_to_spline_basis(extract_spline_quantum_amplitudes(eval_spline.transform(x_tr)))
        b_te_best = amplitudes_to_spline_basis(extract_spline_quantum_amplitudes(eval_spline.transform(x_te)))
        eval_weights = fit_penalized_spline_cdf_solver(b_tr_best, y_tr, penalty=best_penalty)
        test_nll = evaluate_cdf_nll(eval_weights, b_te_best, y_te)

        




        final_spline = SplineTransformer(
            n_knots=best_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        )
        final_spline.fit(x_raw)
        b_all = amplitudes_to_spline_basis(extract_spline_quantum_amplitudes(final_spline.transform(x_raw)))
        final_weights = fit_penalized_spline_cdf_solver(b_all, y_all, penalty=best_penalty)

        x_next = (draws[-WINDOW_LEN:].astype(np.float64) / float(MAX_NUM)).reshape(1, -1)
        b_next = amplitudes_to_spline_basis(extract_spline_quantum_amplitudes(final_spline.transform(x_next)))
        raw_cdf_next = (b_next @ final_weights).__getitem__(0)
        next_combo = inverse_cdf_ordered_decode(raw_cdf_next)

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (knots={best_knots}, pen={best_penalty}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Qiskit Cubic B-Spline Controlled-Amplitude CDF Regressor (1x6 qubita)
Posebnost: Hronoloski B-Spline cvorovi + kontrolisano CRY kolo + zatvorena CDF regresija na simulatoru

CSV: loto7_4718_k90.csv
Validacija NLL (knots=6, pen=5.0): 20.7575
Test NLL: 20.5991
NEXT: [1, x, 13, y, 26, z, 39]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (knots=6, pen=5.0): 20.6915
Test NLL: 20.9601
NEXT: [1, x, 12, y, 27, z, 37]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (knots=6, pen=5.0): 20.6263
Test NLL: 20.8764
NEXT: [2, x, 13, y, 27, z, 39]
"""



"""
• Spline / B-Spline regresije (kubični splajnovi po čvorovima — knots, piecewise polynomials) i regularizovane / lokalno glatke regresije.

• Osnovna ideja: Umesto jednog globalnog polinoma visokog stepena koji oscilira (Runge-ov fenomen) ili čisto linearnog modela, Spline regresija deli domen ulaznih promenljivih na segmente pomoću čvorova (knots) i na svakom segmentu uklapa glatke B-spline bazne funkcije (\(B_{i,k}(x)\)), uz kontrolu glatkoće (Penalized Splines / Ridge regularizacija) kako bi se uhvatile lokalne nelinearne promene bez preprilagođavanja.

• osnovnu ideju kubičnih B-spline baznih funkcija ugrađujem direktno u Qiskit Quantum Spline Amplitude Circuit (1x6 qubita) + Penalized Quantum Kernel / Spline Ridge Solver u zatvorenom obliku (bez L-BFGS-B gradijentnog spuštanja!).
		
1. Hronološki niz prethodnih kola mapira se kroz kubične B-spline bazne funkcije (učene isključivo na prošlim prelazima po Pravilu 10, sa brojem čvorova knots biranim na validaciji).
2. Dobijeni B-spline koeficijenti upravljaju kontrolisanim rotacijama (CRY / CRZ) i multi-kontrolisanim amplitudnim mešanjem na 6 qubita (simulator, rad sa amplitudama).
3. Iz izmerenih kvantnih amplituda rešava se zatvoreni sistem penalizovane splajn regresije (Penalized Quantum Amplitude Spline Ridge) za neprekidnu kumulativnu raspodelu (CDF / uslovne kvantile) svih 39 brojeva, odakle se NEXT kombinacija dobija inverznim preslikavanjem naučene kumulativne splajn raspodele (bez dosadašnjeg backtrack pretraživača i bez zaokruživanja 7 proseka).

• B-spline bazne funkcije po čvorovima omogućavaju kvantnom kolu da lokalno reaguje na promene u različitim delovima opsega (npr. niski vs. visoki brojevi u poslednjim kolima) umesto globalne tanh saturacije.
• Zatvoreno rešenje (Penalized Spline Ridge nad kvantnim amplitudama) potpuno eliminiše scipy.optimize.minimize i sporu iterativnu optimizaciju — izvršava se u jednom matričnom koraku po hiperparametru (čvorovi + kazna glatkoće), što je brzo i deterministički.

Hronološka validacija (izbor broja čvorova splajna i parametra penalizacije glatkoće) i test, zatim završno učenje na celoj istoriji.
"""
