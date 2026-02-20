Sim — agora que você passou o repo eu consegui revisar o código. E a boa notícia é: **você já tem “braços” bem perto do que você descreveu**. O agente já expõe **tools** (`computer`, `browser`, `shell`, `notebook`) e já existe um caminho de **execução de comandos** (via `shell`) com perfil de execução e políticas de segurança. ([GitHub][1])

A parte mais “cara” agora não é criar do zero, e sim **tornar o shell/script algo consistente, configurável e seguro** em *todas* as plataformas (Windows/macOS), e deixar a expansão de tools menos “hard-coded”.

## O que já existe hoje (e onde)

* **Tooling (LLM → ação local):** o `CognitiveCore` define os tools e mapeia a tool-call para o esquema de ação interno. O `shell` vira ação `{"type": "sandbox_shell", "execution":"shell", "cmd": ...}`. ([GitHub][2])
* **Profiles de execução:** `EXECUTION_PROFILE=local_gui|remote_cli|hybrid` controla se GUI/browser/shell aparecem e/ou são bloqueados. ([GitHub][3])
* **Policy/HITL:** existe `PolicyEngine` com regras YAML e decisões que podem bloquear ou exigir confirmação humana. ([GitHub][4])
* **Drivers OS-específicos:**

  * Windows tem driver de shell PowerShell com validação (allowlist, sem redirecionamento, workspace). ([GitHub][5])
  * macOS tem driver de shell simples que roda `subprocess.run(argv, cwd=workspace)` e confia mais na policy. ([GitHub][6])
* **Loop:** o `Orchestrator` executa a ação e injeta `shell_stdout`/`shell_stderr` no histórico para o modelo usar no próximo passo. ([GitHub][7])

Ou seja: **“capacidade de execução de scripts” já é possível via `shell`**, principalmente em `remote_cli`/`hybrid` — mas existem alguns “gargalos” e inconsistências que eu atacaria antes de abrir o leque.

---

## Pontos que eu melhoraria antes de “ligar script execution” geral

### 1) Inconsistência: YAML fala “shell_command”, mas o agente usa “sandbox_shell”

No `safety_rules.yaml` você bloqueia `shell_command`, mas **a ação real** que sai do `CognitiveCore` é `sandbox_shell`. Então, do jeito que está, *esse bloqueio não pega*. ([GitHub][8])

✅ Ajuste prático:

* Trocar no YAML para `blocked_actions: - sandbox_shell` (se a intenção é bloquear shell).
* Ou mapear/normalizar “shell_command” → “sandbox_shell” no `PolicyEngine` para compatibilidade.

### 2) Inconsistência: `.env.example` promete `SHELL_ALLOWED_COMMANDS` no macOS, mas a policy não aplica

O `.env.example` diz que `SHELL_ALLOWED_COMMANDS` “estende defaults” e dá exemplo macOS (`ls,echo,grep,wc,git`). ([GitHub][3])
Mas no `PolicyEngine` do core, a validação do `sandbox_shell` usa um allowlist **hardcoded por caminho absoluto** (`ALLOWED_COMMANDS = {"/bin/ls": ["*"], ...}`) e **não usa** `allowed_shell_basenames` do YAML/env para decidir. ([GitHub][4])

Isso vai confundir quem configurar o `.env` e esperar que funcione no macOS.

✅ Ajuste prático (mínimo e coerente):

* No `PolicyEngine.evaluate()` quando `action_type == "sandbox_shell"`, além do allowlist por path, incorporar:

  * allowlist por basename vindo de `settings.shell_allowed_commands` ou `rules["allowed_shell_basenames"]`
  * e/ou gerar dinamicamente um map `{resolved_path: ["*"]}` a partir dos basenames (com validação do diretório do binário, para não confiar em PATH “esquisito”).

### 3) Duplicidade de allowlist no Windows: policy vs driver

No Windows:

* `WindowsPolicyEngine` valida o comando `sandbox_shell` por token (allowlist) e marca HITL para scripts/destrutivos. ([GitHub][9])
* O `ShellDriver` do Windows tem **outra** allowlist (defaults + env) e também bloqueia operadores e paths. ([GitHub][5])

Isso é bom (defesa em profundidade), **mas** também dá para cair num estado “policy permite, driver bloqueia” dependendo de como a pessoa edita YAML vs env.

✅ Ajuste prático:

* Escolher **uma fonte de verdade** (idealmente `Settings`/env) e fazer ambos consumirem a mesma lista.
* Ou, se YAML for a fonte, o driver também deveria ler YAML (ou receber do policy engine).

### 4) Exposição do tool “shell” mesmo quando `ENABLE_SHELL=false`

Hoje, o `CognitiveCore` habilita `shell` com base no profile, e o driver faz “dry-run” quando `ENABLE_SHELL=false`. Isso pode levar o modelo a insistir em shell e “perder passos” sem executar nada. ([GitHub][2])

✅ Ajuste prático:

* Fazer `_tool_enabled_map()` considerar **profile + enable_shell**, ou pelo menos acrescentar no system prompt uma frase explícita tipo “shell está em dry-run”.

---

## Sobre “dar capacidade de execução de scripts”: eu faria assim

### Caminho A (rápido): “script execution” como subconjunto do `shell`

Isso é o mínimo viável e encaixa no desenho atual:

* Você **não cria** tool nova.
* Você só amplia allowlists e adiciona HITL/guardrails.

Recomendação:

1. **Criar uma regra clara para script execution**:

   * Se comando contiver `python`, `node`, `.sh`, `.py`, `.js`, etc:

     * por padrão: `hitl_required=True`
     * e/ou exigir uma flag tipo `ALLOW_SCRIPT_EXEC=true` no settings.
2. **Workspace obrigatório**:

   * Só permitir executar scripts que estejam dentro de `SHELL_WORKSPACE_ROOT`.
3. **Sem rede (honestamente)**:

   * “Sem rede” via shell é **difícil de garantir** se você liberar Python/Node (eles podem abrir socket). Então:

     * trate isso como “alto risco” e gateie com HITL
     * ou rode em ambiente realmente isolado (container/VM/sandbox OS) se isso for um requisito forte.

> Na prática: “habilitar python” = abrir uma porta enorme. Eu só faria com HITL ou com isolamento real.

### Caminho B (melhor UX e mais seguro): tool novo `script`

A vantagem é que você consegue dar ao LLM uma interface *mais restrita* do que um shell genérico, e ainda assim poderosa para tarefas de automação.

Proposta de tool `script` (bem objetiva):

* `script.write` → cria/atualiza um arquivo em `.agent_shell/` (somente paths relativos)
* `script.run` → roda com runtime/output limit, sempre no workspace
* `script.read` → lê arquivo do workspace

E o mais importante: você consegue implementar políticas tipo:

* só permitir runtime de no máximo X segundos
* só permitir executar extensões específicas
* exigir HITL quando:

  * script contiver import de `socket`, `subprocess`, `os`, etc
  * ou quando tentar acessar path fora do workspace

Isso resolve o problema clássico: **“shell virou faca de chef”**, mas você quer uma **ferramenta de bancada**.

---

## “Expandir os tools”: eu sugiro uma refatoração pequena que paga muito

Hoje, para adicionar tool você mexe direto no `CognitiveCore` (schema + parsing + mapping). ([GitHub][2])
Se você quer crescer isso, eu criaria um mini “plugin registry”:

* `ToolSpec` (schema OpenAI)
* `ToolMapper` (args → action dict)
* `ToolExecutor` (execution path / driver)
* `ToolPolicy` (policy checks adicionais opcionais)

Isso deixa adicionar tools novos (ex.: `workspace`, `script`, `http_download`, `pdf_extract`) bem mais limpo.

---

## Se você quiser uma lista de “quick wins” (alta alavancagem)

1. **Corrigir o mismatch `shell_command` vs `sandbox_shell` no YAML** (impacto imediato). ([GitHub][8])
2. **Fazer `SHELL_ALLOWED_COMMANDS` funcionar no macOS** (docs batendo com código). ([GitHub][3])
3. **Unificar allowlist do Windows (policy/driver)** para evitar “permite aqui, bloqueia ali”. ([GitHub][5])
4. **Não expor `shell` tool quando `ENABLE_SHELL=false`** (ou avisar muito explicitamente). ([GitHub][3])
5. **Adicionar HITL padrão para qualquer coisa “script-like” no macOS** (hoje o Windows tem heurística forte; macOS fica mais “solto” se você ampliar allowlist). ([GitHub][9])

---

[1]: https://raw.githubusercontent.com/lacassef/computeruse/master/README.md "https://raw.githubusercontent.com/lacassef/computeruse/master/README.md"
[2]: https://raw.githubusercontent.com/lacassef/computeruse/master/cua_agent/agent/cognitive_core.py "https://raw.githubusercontent.com/lacassef/computeruse/master/cua_agent/agent/cognitive_core.py"
[3]: https://raw.githubusercontent.com/lacassef/computeruse/master/.env.example "https://raw.githubusercontent.com/lacassef/computeruse/master/.env.example"
[4]: https://raw.githubusercontent.com/lacassef/computeruse/master/cua_agent/policies/policy_engine.py "https://raw.githubusercontent.com/lacassef/computeruse/master/cua_agent/policies/policy_engine.py"
[5]: https://raw.githubusercontent.com/lacassef/computeruse/master/windows_cua_agent/drivers/shell_driver.py "https://raw.githubusercontent.com/lacassef/computeruse/master/windows_cua_agent/drivers/shell_driver.py"
[6]: https://raw.githubusercontent.com/lacassef/computeruse/master/macos_cua_agent/drivers/shell_driver.py "https://raw.githubusercontent.com/lacassef/computeruse/master/macos_cua_agent/drivers/shell_driver.py"
[7]: https://raw.githubusercontent.com/lacassef/computeruse/master/cua_agent/orchestrator/orchestrator.py "https://raw.githubusercontent.com/lacassef/computeruse/master/cua_agent/orchestrator/orchestrator.py"
[8]: https://raw.githubusercontent.com/lacassef/computeruse/master/cua_agent/policies/safety_rules.yaml "https://raw.githubusercontent.com/lacassef/computeruse/master/cua_agent/policies/safety_rules.yaml"
[9]: https://raw.githubusercontent.com/lacassef/computeruse/master/windows_cua_agent/policies/windows_policy_engine.py "https://raw.githubusercontent.com/lacassef/computeruse/master/windows_cua_agent/policies/windows_policy_engine.py"
