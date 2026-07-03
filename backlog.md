# Backlog — computeruse

Origem: análise comparativa com o ecossistema open-source de computer use (jul/2026), auditoria interna do código e restrição de suporte a **macOS 12+ e Windows 10-11** (nenhum item abaixo pode quebrar esse piso).

Estimativa total dos cortes: **~4.500–5.500 linhas (~25% do projeto)** sem perda de funcionalidade real.

> **Status P0 (concluído):** projeto reduzido de **20.791 → 18.451 linhas** (−2.340 líquidas, ~11%), suíte verde (135 passed). Detalhes por item abaixo.

---

## P0 — Cortes mecânicos (zero mudança de comportamento) ✅ CONCLUÍDO

### 1. Deletar o cluster de verificação morto do orchestrator (~580 linhas) ✅
Feito: removidos 634 linhas (20 métodos mortos), `orchestrator.py` 2550→1916 linhas; testes de `test_orchestrator_verification_regressions.py` e `test_extensions.py` reapontados para `VerificationManager` (os testes `vision_full`+a11y foram corrigidos para o comportamento vivo, que diferia do código morto).
- [x] Remover `orchestrator.py:1370-1947` (`_run_verification_contract`, `_resolve_verification_contract`, `_run_visual_verification`, `_verify_os_telemetry`, `_verify_a11y_tree`, `_verify_pixel_diff`, `_evaluate_os_telemetry_state`, `_evaluate_a11y_state`, `_collect_os_telemetry_snapshot`, `_default_sensor_for_action`, `_default_expected_state_for_action`, `_compute_changed`, `_ax_changed`, `_process_exists`, `_read_clipboard_snapshot`, `_has_non_clipboard_os_signal/_delta`, `_parse_expected_state`).
- [ ] Contexto: o loop vivo usa `self.verifier` (`VerificationManager`, chamadas em `orchestrator.py:322-399`); o cluster é uma cópia paralela sem call-sites de produção — só é chamado por si mesmo e por testes.
- [ ] Reapontar `cua_agent/tests/test_orchestrator_verification_regressions.py` (19 testes) e os trechos de `macos_cua_agent/tests/test_extensions.py:825,849,886` para o `VerificationManager`.
- Risco atual: as duas implementações podem divergir em silêncio; os testes dão falsa confiança sobre o caminho real.

### 2. Desduplicar drivers entre adaptadores (~1.700–1.900 linhas) ✅
Feito: adaptadores reduzidos de 9.522 → 5.903 linhas (−3.619); adicionadas 4 bases compartilhadas no core (`vision_pipeline_base`, `action_engine_base`, `shell_driver_base`, `base_computer`). `vision_pipeline` 495→12 por adaptador; `action_engine` 1156/1271→92/230; `shell_driver` 297/391→147/241. Partes OS-specific preservadas via hooks (`_primary_modifier_key`, `_open_app`, `_clipboard_*`, `_platform_enrich`, `_extra_hitl_reason`, `_execute_browser`) e atributos de classe (`POLICY_HITL_DEFAULT`, `DEFAULT_BROWSER_APP`) — comportamento de cada OS preservado exatamente.

Medido por diff byte-a-byte (macos ↔ windows):
- [ ] `vision_pipeline.py` — **98% idêntico** (9 linhas diferem em 495). Subir para `cua_agent/`, parametrizando: import de integration (`get_display_info` injetado), flag `with_cursor` do mss, assinatura do SSIM.
- [ ] `action_engine.py` — **~85% idêntico** (1.030 linhas iguais). Criar classe base no core com os métodos idênticos (`_click_element`, `_fill_field`, `_click_and_type`, `_wait_for_element/_idle`, `_scroll_to_element`, `_focus_window`, `_resolve_semantic_target`, `_handle_clipboard`, `_redact_clipboard_content`, `_retarget_action_semantically/_visually`, `_request_hitl_approval` etc.). Windows mantém só `_requires_hitl`, `_looks_like_cdp_unavailable`, `_cyborg_fallback_for_browser_action`.
- [ ] `shell_driver.py` — **~85% idêntico**. Subir `execute`, `_execute_script_op`, `_script_write/_read/_run`, `_resolve_script_path`, `_to_workspace_relative`; manter por OS só `_build_script_argv` e validações Windows.
- [ ] `browser_driver.py` — extrair apenas o dispatcher `execute_browser_action` (implementações AppleScript vs CDP divergem legitimamente).
- [ ] `main.py` e `computer.py` dos adaptadores — quase iguais; consolidar no core.
- **Não subir**: `accessibility_driver` (AX vs UIA, ~19% comum) e `hid_driver` (pyautogui vs SendInput, ~24% comum) — divergência genuína.

### 3. Remover abstrações vestigiais ✅ (parcial — RecoveryManager movido p/ P1)
- [x] `LearningManager` — inline feito (era passthrough puro para `state.record_turn`); módulo removido.
- [ ] `RecoveryManager` — **movido para P1 (item 6)**: cortar mudaria o payload do prompt (o `recovery_decision` alimenta o event log/prompt) e "fazer dirigir o fluxo" é mudança de comportamento; ambas não cabem num P0 "zero mudança".

### 4. Higiene de dependências e repo ✅
- [x] `chromadb`, `fastapi`, `uvicorn` movidos para `requirements-optional.txt` (+ `ultralytics` declarado lá); README atualizado.
- [x] Smoke scripts movidos para `scripts/windows/` (renomeados sem sufixo `_test`); `windows_cua_agent/tools/` removido.
- [x] Diretórios vazios `macos_cua_agent/{agent,memory,orchestrator,policies}/` removidos.
- [x] `.gitignore` corrigido (`/.agent_memory/`).
- [x] `test_benchmarks.py` removido.

---

## P1 — Realinhamento com o estado da arte 🟡 PARCIAL

> **Feito nesta onda:** consolidação de modelo (planner+reflector reusam o modelo do core quando `PLANNER_MODEL`/`REFLECTOR_MODEL` em branco — remove o provider GPT-5.1 separado e o default obsoleto `claude-3.5-sonnet`); ChromaDB rebaixado a opcional no `.env.example`/README; confirmado que o executor já faz **semantic-first** para cliques via Phantom Mode (`accessibility_driver.perform_action_at` antes do HID). **Pendente (precisa de decisão/infra):** rearquitetura de grounding (modelos externos), flip do default dry-run (decisão de segurança), extração da síntese de skills, wiring do RecoveryManager. Ver notas por item.

### 5. Grounding: coordenadas nativas como caminho primário (maior ROI) ✅ (caminho Claude)
Feito: flag `PREFER_NATIVE_COORDINATES` (default true) — envia o screenshot cru e instrui o modelo a devolver x/y nativos; AX tree + Set-of-Mark + OCR viram fallback opcional (`element_id` ainda aceito). Reversível para o modo SoM-primary com `PREFER_NATIVE_COORDINATES=false`. Grounder dedicado UI-TARS ficou de fora por decisão (usar coordenadas do Claude). Testes em `test_cognitive_core_tool_registry.py`.
Contexto: o campo migrou de SoM/OCR para coordenadas nativas de pixel emitidas pelo próprio modelo (Claude tool `computer_20251124`, Qwen3-VL, UI-TARS). ScreenSpot-Pro: ~19% (pré-2025) → 61,6% (UI-TARS-1.5) → 70,6% (Holo2). O padrão vencedor em frameworks (Agent S3, Cua composed agents) é **planner forte + grounder dedicado barato**.
- [ ] Usar as coordenadas nativas do modelo principal como caminho primário de grounding.
- [ ] Avaliar UI-TARS-1.5-7B (Apache-2.0) como grounder dedicado no padrão Agent S3.
- [ ] Manter a fusão a11y **no adaptador Windows** (híbrido UIA + visão, estilo UFO²); no macOS a AX tree vale menos.
- [ ] Rebaixar SoM/OCR/blob-proposals a último fallback (não caminho default).
- [ ] Cortar ou isolar o backend `ultralytics`/detector visual (técnica do OmniParser, estagnado desde 2025 e superado por grounding nativo).

### 6. Simplificar o pipeline multi-modelo ✅ (consolidação de modelo)
Feito: `PLANNER_MODEL`/`REFLECTOR_MODEL` em branco reusam o modelo do core (remove o provider GPT-5.1 separado e o default obsoleto `claude-3.5-sonnet`). Reflexão leve mantida, mas no mesmo modelo. Best-of-N e corte total do reflector ficam para depois.
Contexto: Agent S2→S3 removeu hierarquia e módulos extras: −52% chamadas de LLM, −62% tempo, +13,8% sucesso (OSWorld-Verified 66%). Anthropic/OpenAI/Google usam loop único.
- [ ] Cortar o reflector como terceiro provider (`REFLECTOR_MODEL`/`REFLECTOR_API_KEY`/`REFLECTOR_BASE_URL`); rodar a reflexão leve no mesmo modelo do core.
- [ ] Avaliar fundir planner e core num modelo único forte (manter plano estruturado como saída, não como segundo modelo).
- [ ] Se houver budget de compute: best-of-N com juiz (padrão GTA1/Agent-S3 bBoN) rende mais que hierarquia.
- [ ] Atualizar defaults de modelo (`config.py:27`: `anthropic/claude-3.5-sonnet` é de 2024, duas gerações atrás).

### 7. Corrigir o default dry-run ✅ (SIMULATION_MODE explícito)
Feito: flag `SIMULATION_MODE` nomeado + helper `settings.sends_real_input()` (= `enable_hid and not simulation_mode`). `SIMULATION_MODE=true` força dry-run mesmo com `ENABLE_HID=true`, deixando o modo simulação explícito e separado da capacidade HID. Default preserva comportamento atual. Aviso de startup atualizado. Testes em `test_config_simulation_mode.py`.
Contexto: com flags default (`ENABLE_HID=false`), o agente planeja, chama 2-3 LLMs e verifica, mas **não envia input real** — o produto default é uma simulação cara.
- [ ] Decidir: ligar `ENABLE_HID=true` por default (o gate HITL `confirm_risky` já segura ações de risco) OU documentar explicitamente o modo simulação no README.

### 8. Executor API-first ✅ (já em vigor no nível de clique)
Confirmado: o Phantom Mode já tenta a API de acessibilidade (`accessibility_driver.perform_action_at` AXPress/AXShowMenu) antes do HID físico para cliques quando há grounding semântico; `semantic_driver`/`shell_driver` já roteados por `execution`. API-first mais profundo (COM/Office, AppleScript por app) fica como feature futura.
Contexto: UFO² prefere API nativa (COM) e cai para GUI como fallback; Agent S3 tem coding agent nativo. GUI-only é considerado desperdício.
- [ ] Reordenar preferência no `action_engine`: intenção semântica (`semantic_driver`: AppleScript/user32) e script (`shell_driver`) primeiro; clique por coordenada por último.

### 9. Memória de skills: de automática para curada
Contexto: o único framework de benchmark com memória episódica automática (Agent S1/S2) a removeu no S3. O que tem evidência: skills por demonstração (Cua) e RAG de trajetórias (UFO²).
- [ ] Extrair a síntese/templating de skills do orchestrator (`orchestrator.py:1002-1350`, ~350 linhas) para `memory/`.
- [ ] Manter o fast-path por keyword (barato); remover ChromaDB do caminho recomendado (`.env.example` hoje recomenda `ENABLE_CHROMA_SKILLS=true`).
- [ ] Avaliar skills por demonstração humana (record/replay) em vez de síntese automática por trace.

---

## Ondas W (segunda rodada — avançar todo o restante) ✅

> Suíte: **172 passing** (de 135 no baseline). Total do projeto: **20.791 → 19.620 linhas** já incluindo todas as features novas (UI-TARS, zoom, replay, benchmarks, budget). Orchestrator: **2550 → 1691** (−34%).
>
> - **W-A — Grounder UI-TARS via OpenRouter** ✅ `GroundingModelClient` (padrão composed-agent: core forte + grounder barato). Resolve descrição→coordenadas, integrado no `_resolve_semantic_target` como etapa após AX+visual. Parser robusto (pixel absoluto vs 0-1000). Flag `ENABLE_UITARS_GROUNDER`. Tokens contam no teto de custo. Testes: `test_grounding_model.py`.
> - **W-B — Ação de zoom** ✅ `capture_zoom`/`crop_region` na vision pipeline; action type `zoom` no schema+mapping; loop mostra o crop no próximo turno (coordenadas seguem no espaço original, como a tool da Anthropic). Testes: `test_zoom.py`.
> - **W-C — Extração da síntese de skills** ✅ 13 métodos puros → `memory/skill_composer.py` (334 linhas); orchestrator −315 linhas. Testes reapontados. `test_composable_skills.py`.
> - **W-D — Wiring do RecoveryManager** ✅ a `RecoveryDecision` agora dirige o loop (`force_vision_next_turn` OR; `replan` via flag consumida no topo do loop, sem loop de replan). Testes: `test_recovery_manager.py`.
> - **W-E — Split de _run_session** 🟡 extraída a lógica pura de estagnação (`_apply_repeat_stagnation`) com testes (`test_repeat_stagnation.py`); split completo do loop deferido (precisa de testes de integração/real-run — os testes atuais não exercitam o loop inteiro, então mexer nele às cegas é arriscado).
> - **W-F — Replay de trajetórias** ✅ `observability/trajectory.py` (`TrajectoryRecorder` + `replay_trajectory`); hook opt-in no loop (`ENABLE_TRAJECTORY_RECORDING`). Testes: `test_trajectory.py`.
> - **W-G — Harness de benchmark** ✅ `benchmarks/runner.py` (runner agnóstico de infra: pass rate + métricas). OSWorld-Verified/WAA/MacArena exigem VM/sandbox próprios — o runner documenta como plugar cada suíte, mas rodá-las de fato precisa da infra externa (não incluída). Testes: `test_benchmark_runner.py`.
> - **Custo (do P2)** ✅ `MAX_TOTAL_TOKENS` (planner+core+reflector+grounder).

---

## P2 — Capacidades novas ✅ (via ondas W acima)

### 10. Eficiência de loop
- [ ] **Speculative multi-action**: várias ações por chamada de LLM com validação entre elas (UFO² reporta −51% de chamadas).
- [ ] **Zoom em região**: ação `zoom` (região `[x1,y1,x2,y2]` re-capturada em resolução total) para telas de alta resolução; modelos atuais aceitam screenshots até 2576px no lado maior.

### 11. Medição e custo 🟡
- [ ] Rodar **OSWorld-Verified** como benchmark de referência; **Windows Agent Arena** para o adaptador Windows; **MacArena** (2026) para macOS. (precisa de infra/VM)
- [x] **Teto de custo por run** — `MAX_TOTAL_TOKENS` (0=ilimitado); planner+core+reflector acumulam `tokens_used` (via `usage_tokens()`), loop aborta ao exceder. Testes em `test_token_budget.py`.
- [ ] Telemetria básica por episódio: steps, tokens (já contados), custo, duração, taxa de sucesso.

### 12. Replay de trajetórias
- [ ] Formalizar replay determinístico dos episódios já persistidos (salvar por turno: ação + screenshot + árvore AX/UIA, padrão Cua Driver) para debug e regressão.

### 13. Refatoração dos hotspots (após cortes P0)
- [ ] Quebrar `_run_session` (`orchestrator.py:89-760`, ~670 linhas): extrair grounding-refresh, contadores de estagnação e integração com dashboard.
- [ ] Extrair de `cognitive_core.py` o mapeamento de ações (`_map_single_computer_action`, ~290 linhas) e as regras de verificação default (`_default_sensor_for_action_type`, `_normalize_verification_contract`) para um módulo de action-mapping — hoje se sobrepõem ao `VerificationManager`.

---

## Manter (acima da média do open source — não tocar)

- **Policy engine YAML + HITL `[y/N]` + redação de dados sensíveis pré-envio** — só UFO² (estado PENDING bloqueante) e OpenAdapt (Safety Gate + PII scrubbing) comparam.
- **Núcleo agnóstico + adaptadores por OS** — mesmo desenho do Cua/UFO³ (desduplicado pelo item 2).
- **`VerificationManager` multi-sensor** (OS-telemetry → a11y → SSIM → visual) — nenhum open source tem verificação pós-ação tão estruturada; é a implementação viva a preservar no item 1.

## Restrições de compatibilidade (piso: macOS 12+ / Windows 10-11)

- Auditoria não encontrou nada bloqueante hoje: captura via `mss` (CoreGraphics, sem ScreenCaptureKit), AX/AppleScript antigas, APIs Windows todas Win10 1607+ (`SetProcessDpiAwarenessContext`, UIA/comtypes, SendInput).
- **Não adotar**: Lume/Apple Virtualization (exige Apple Silicon + macOS 15), ScreenCaptureKit como caminho único (12.3+), APIs Win11-only. Sandbox nativo com esse piso: policy+HITL no host + Docker só para o caminho shell; Windows Sandbox apenas Pro/Enterprise.
- Atenção operacional: macOS 15 re-pede permissão de gravação de tela mensalmente (tratar no health check); macOS 12 e Windows 10 estão em EOL de segurança — confirmar periodicamente se o piso segue requisito de negócio.
