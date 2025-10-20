-- emerald_ai.lua
-- Single-file self-learning agent for Pokémon Emerald (GBA)
-- Target: mGBA or BizHawk Lua environments (emulator-agnostic wrappers)
-- Runs entirely inside emulator; persists its own learning to disk.

-------------------------------------------------------------------------------
-- Configuration
-------------------------------------------------------------------------------
local CONFIG = {
  save_dir = '.',
  qtable_filename = 'emerald_ai.qtable.tsv',
  calib_filename = 'emerald_ai.calib.tsv',
  meta_filename = 'emerald_ai.meta.tsv',
  log_filename = 'emerald_ai.log',

  -- RL parameters
  epsilon_start = 0.20,       -- initial exploration rate
  epsilon_min = 0.02,         -- minimum exploration rate
  epsilon_decay_frames = 2e6, -- frames over which epsilon decays to epsilon_min
  learning_rate = 0.20,
  discount_factor = 0.995,

  -- Control loop
  frames_per_action = 10,     -- how long to hold a button for each action
  frames_between_steps = 2,   -- idle frames between actions
  autosave_interval_frames = 2000,

  -- Novelty / reward
  novelty_reward = 0.5,       -- reward for first time seeing a coarse cell
  stuck_penalty = -0.5,
  small_move_reward = 0.05,   -- tiny reward for movement (encourage walking)
  -- Phase/interaction shaping
  phase_stagnant_threshold_steps = 20, -- steps without movement to consider as interaction
  interaction_exit_reward = 1.0,       -- reward for leaving long interaction/battle/menu phases
  long_interaction_penalty = -0.2,     -- periodic penalty during very long interactions
  max_interaction_steps_without_exit = 200, -- after which we inject exploration
  macro_hold_frames = 6,               -- default hold for macro button taps

  -- Higher-level shaping
  catch_attempt_window_frames = 1200,  -- ~20s window to attribute catch
  catch_reward = 5.0,
  warp_reward = 1.0,
  new_map_reward = 1.0,
  badge_reward = 20.0,
  warp_distance_threshold = 24,        -- XY manhattan distance to count as warp/transition

  -- Savestates / episodes
  savestate_enable = true,
  savestate_slots = 5,
  savestate_save_every_new_cells = 100,
  no_novelty_reset_frames = 6000,

  -- Coordinate detection
  calib_move_frames = 18,
  calib_scan_regions = {
    {name = 'EWRAM', start_addr = 0x02000000, size = 0x40000}, -- 256KB
    {name = 'IWRAM', start_addr = 0x03000000, size = 0x08000}, -- 32KB
  },
  calib_try_directions = {'Up', 'Down', 'Left', 'Right'},
  calib_candidate_max = 64,    -- cap candidates for safety

  -- Coarse grid for novelty map
  coarse_cell_size = 4,        -- pixels per cell
  novelty_map_capacity = 100000,

  -- Optional manual memory hints (hex strings like '0x0203ABCD' or numbers)
  hints = {
    pokedex_caught_addr = nil,
    map_id_addr = nil,
    badge_flags_addr = nil,
  },

  -- Logging verbosity
  log_every_n_steps = 200,
}

-------------------------------------------------------------------------------
-- Emulator Abstraction Layer (BizHawk/mGBA compatible where possible)
-------------------------------------------------------------------------------
local EMU = {
  name = 'unknown',
  has_memory_domains = false,
  memory_domain = nil,
  joypad_port = 1,
}

local function safe_pcall(fn, ...)
  local ok, res1, res2, res3 = pcall(fn, ...)
  if ok then return true, res1, res2, res3 end
  return false, nil
end

local function detect_emulator()
  -- Try to detect functions specific to BizHawk or mGBA
  if client and type(client.getversion) == 'function' then
    EMU.name = 'bizhawk'
  elseif emu and type(emu.framecount) == 'function' and package and package.cpath and not client then
    EMU.name = 'mgba' -- heuristic
  else
    EMU.name = 'unknown'
  end

  -- Memory domain capabilities (BizHawk)
  if memory and type(memory.usememorydomain) == 'function' then
    EMU.has_memory_domains = true
  end

  -- Select best default domain if available (BizHawk)
  if EMU.has_memory_domains then
    local ok = pcall(function() memory.usememorydomain('System Bus') end)
    if ok then EMU.memory_domain = 'System Bus' end
  end
end

detect_emulator()

-------------------------------------------------------------------------------
-- Logging and File Utilities
-------------------------------------------------------------------------------
local LOG = {}
local function now_frame()
  if emu and emu.framecount then
    return emu.framecount()
  end
  return 0
end

local function write_file(path, text)
  local f, err = io.open(path, 'w')
  if not f then return false, err end
  f:write(text)
  f:close()
  return true
end

local function append_file(path, text)
  local f, err = io.open(path, 'a')
  if not f then return false, err end
  f:write(text)
  f:close()
  return true
end

local function read_file(path)
  local f = io.open(path, 'r')
  if not f then return nil end
  local content = f:read('*a')
  f:close()
  return content
end

local function log_line(msg)
  local line = string.format('[%d][%s] %s\n', now_frame(), EMU.name, msg)
  append_file(CONFIG.save_dir .. '/' .. CONFIG.log_filename, line)
end

-------------------------------------------------------------------------------
-- Joypad Input Wrappers
-------------------------------------------------------------------------------
local function joypad_set(buttons_table)
  -- Both BizHawk and mGBA provide joypad.set(table[, port])
  if joypad and type(joypad.set) == 'function' then
    local ok = pcall(function()
      if EMU.name == 'bizhawk' then
        -- BizHawk typically uses labels like 'P1 Up'. joypad.set takes a table keyed by button names.
        joypad.set(EMU.joypad_port, buttons_table)
      else
        joypad.set(buttons_table)
      end
    end)
    if ok then return end
  end

  -- Fallback: try to synthesize via input API if available (rare)
end

local function press_buttons(button_names, hold_frames)
  hold_frames = hold_frames or CONFIG.frames_per_action
  local buttons = {}
  for _, name in ipairs(button_names) do buttons[name] = true end
  joypad_set(buttons)
  for _ = 1, hold_frames do
    emu.frameadvance()
  end
  joypad_set({})
  for _ = 1, CONFIG.frames_between_steps do emu.frameadvance() end
end

local function press_button(name, hold_frames)
  press_buttons({name}, hold_frames)
end

-- Short tap helper for macros
local function tap(name, hold_frames)
  press_button(name, hold_frames or CONFIG.macro_hold_frames)
end

-- Run a simple macro: sequence of button names or {name, frames}
local function run_macro(seq)
  for _, step in ipairs(seq) do
    if type(step) == 'table' then
      tap(step[1], step[2])
    else
      tap(step)
    end
  end
end

-------------------------------------------------------------------------------
-- Memory Wrappers (endianness-aware where possible)
-------------------------------------------------------------------------------
local function read_u8(addr)
  if memory and memory.read_u8 then return memory.read_u8(addr) end
  if memory and memory.readbyte then return memory.readbyte(addr) end
  if memory and memory.read8 then return memory.read8(addr) end
  return 0
end

local function read_u16_le(addr)
  if memory and memory.read_u16_le then return memory.read_u16_le(addr) end
  if memory and memory.readword then return memory.readword(addr) end
  -- manual little-endian compose
  local lo = read_u8(addr)
  local hi = read_u8(addr + 1)
  return lo + hi * 256
end

local function read_u32_le(addr)
  if memory and memory.read_u32_le then return memory.read_u32_le(addr) end
  if memory and memory.readdword then return memory.readdword(addr) end
  local b0 = read_u8(addr)
  local b1 = read_u8(addr + 1)
  local b2 = read_u8(addr + 2)
  local b3 = read_u8(addr + 3)
  return b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
end

local function read_bytes(addr, count)
  if memory and memory.read_bytes_as_array then
    return memory.read_bytes_as_array(addr, count)
  end
  -- Fallback: byte-by-byte (slower)
  local arr = {}
  for i = 0, count - 1 do arr[i + 1] = read_u8(addr + i) end
  return arr
end

-------------------------------------------------------------------------------
-- Persistent Storage (Q-table, Calibration, Meta)
-------------------------------------------------------------------------------
local META = {
  total_steps = 0,
  total_episodes = 0,
}

local function load_meta()
  local path = CONFIG.save_dir .. '/' .. CONFIG.meta_filename
  local content = read_file(path)
  if not content then return end
  for line in content:gmatch('[^\n]+') do
    local k, v = line:match('^(%w+)\t(.+)$')
    if k and v then
      if tonumber(v) then META[k] = tonumber(v) else META[k] = v end
    end
  end
end

local function save_meta()
  local path = CONFIG.save_dir .. '/' .. CONFIG.meta_filename
  local lines = {}
  for k, v in pairs(META) do
    table.insert(lines, string.format('%s\t%s', k, tostring(v)))
  end
  write_file(path, table.concat(lines, '\n'))
end

-- Minimal tab-separated persistence for Q-table
local QTABLE = {
  -- key: state_hash .. '\t' .. action_name
  -- value: { q = number, n = int }
}

local function q_key(state_hash, action_name)
  return state_hash .. '\t' .. action_name
end

local function load_qtable()
  local path = CONFIG.save_dir .. '/' .. CONFIG.qtable_filename
  local content = read_file(path)
  if not content then return end
  local loaded = 0
  for line in content:gmatch('[^\n]+') do
    local shash, action, qstr, nstr = line:match('^(.+)\t([^\t]+)\t([^\t]+)\t([^\t]+)$')
    if shash and action and qstr and nstr then
      local key = q_key(shash, action)
      QTABLE[key] = { q = tonumber(qstr) or 0.0, n = tonumber(nstr) or 0 }
      loaded = loaded + 1
    end
  end
  log_line(string.format('Loaded Q-table entries: %d', loaded))
end

local function save_qtable()
  local path = CONFIG.save_dir .. '/' .. CONFIG.qtable_filename
  local lines = {}
  for key, entry in pairs(QTABLE) do
    local shash, action = key:match('^(.*)\t([^\t]+)$')
    table.insert(lines, string.format('%s\t%s\t%.6f\t%d', shash, action, entry.q, entry.n))
  end
  write_file(path, table.concat(lines, '\n'))
end

-------------------------------------------------------------------------------
-- Calibration + Feature Detection Persistence
-------------------------------------------------------------------------------
local CALIB = {
  player_x_addr = nil,
  player_y_addr = nil,
  addr_domain = 'auto',
  caught_counter_addr = nil,   -- optional autodetected Pokédex caught counter (u16)
  map_id_addr = nil,           -- optional autodetected map id (u16)
  badge_flags_addr = nil,      -- optional autodetected badge flags (u16/32)
}

local function save_calib()
  local path = CONFIG.save_dir .. '/' .. CONFIG.calib_filename
  local lines = {}
  for k, v in pairs(CALIB) do
    if type(v) == 'number' then
      table.insert(lines, string.format('%s\t0x%08X', k, v))
    else
      table.insert(lines, string.format('%s\t%s', k, tostring(v)))
    end
  end
  write_file(path, table.concat(lines, '\n'))
end

local function parse_hex_or_num(s)
  if not s or s == 'nil' then return nil end
  if type(s) == 'number' then return s end
  local hex = s:match('^0x([0-9a-fA-F]+)$')
  if hex then return tonumber(hex, 16) end
  return tonumber(s)
end

local function load_calib()
  local path = CONFIG.save_dir .. '/' .. CONFIG.calib_filename
  local content = read_file(path)
  if not content then return end
  for line in content:gmatch('[^\n]+') do
    local k, v = line:match('^(%w+)\t(.+)$')
    if k and v then
      local num = parse_hex_or_num(v)
      if num ~= nil then CALIB[k] = num else CALIB[k] = v end
    end
  end
end

-------------------------------------------------------------------------------
-- Region Snapshots and XY Calibration
-------------------------------------------------------------------------------
local function snapshot_region(region)
  local bytes = read_bytes(region.start_addr, region.size)
  return bytes
end

local function diff_candidates(before, after)
  -- Return list of addresses where 16-bit little-endian values changed modestly
  local candidates = {}
  local limit = CONFIG.calib_candidate_max
  local size = math.min(#before, #after)
  for i = 1, size - 1, 2 do
    local lo_b, hi_b = before[i], before[i + 1]
    local lo_a, hi_a = after[i], after[i + 1]
    if lo_b and hi_b and lo_a and hi_a then
      local vb = lo_b + 256 * hi_b
      local va = lo_a + 256 * hi_a
      local delta = va - vb
      -- Heuristic: coordinates change by small deltas when we walk
      if delta ~= 0 and math.abs(delta) <= 8 then
        table.insert(candidates, { offset = i - 1, delta = delta })
        if #candidates >= limit then break end
      end
    end
  end
  return candidates
end

local function try_find_xy_in_region(region, dir)
  -- Take baseline
  local base = snapshot_region(region)
  -- Induce movement
  press_button(dir, CONFIG.calib_move_frames)
  local moved = snapshot_region(region)
  -- Undo movement (best-effort)
  local reverse = ({Up='Down', Down='Up', Left='Right', Right='Left'})[dir]
  if reverse then press_button(reverse, CONFIG.calib_move_frames) end
  local cands = diff_candidates(base, moved)
  return cands
end

local function verify_xy_candidates(region, x_offsets, y_offsets)
  -- Move Right: expect X to increase, Y ~ stable
  local base = snapshot_region(region)
  press_button('Right', CONFIG.calib_move_frames)
  local after_right = snapshot_region(region)
  press_button('Left', CONFIG.calib_move_frames)

  local function val_at(arr, off)
    local lo = arr[off + 1]
    local hi = arr[off + 2]
    return lo + 256 * hi
  end

  local function avg_delta(offsets, a, b)
    if #offsets == 0 then return 0 end
    local sum, n = 0, 0
    for _, off in ipairs(offsets) do
      local vb = val_at(b, off)
      local va = val_at(a, off)
      sum = sum + (va - vb)
      n = n + 1
    end
    return sum / math.max(n, 1)
  end

  local dx = avg_delta(x_offsets, after_right, base)
  local dy = avg_delta(y_offsets, after_right, base)

  -- Heuristic: significant dx, near-zero dy
  if math.abs(dx) >= 1 and math.abs(dy) <= 1 then return true end
  return false
end

local function auto_calibrate_xy()
  log_line('Calibration started')
  for _, region in ipairs(CONFIG.calib_scan_regions) do
    local dir_cands = {}
    for _, dir in ipairs(CONFIG.calib_try_directions) do
      dir_cands[dir] = try_find_xy_in_region(region, dir)
    end
    -- Aggregate: vertical moves (Up/Down) likely Y; horizontal (Left/Right) likely X
    local y_offsets_map, x_offsets_map = {}, {}
    for _, d in ipairs({'Up', 'Down'}) do
      for _, c in ipairs(dir_cands[d] or {}) do y_offsets_map[c.offset] = (y_offsets_map[c.offset] or 0) + 1 end
    end
    for _, d in ipairs({'Left', 'Right'}) do
      for _, c in ipairs(dir_cands[d] or {}) do x_offsets_map[c.offset] = (x_offsets_map[c.offset] or 0) + 1 end
    end
    local y_offsets, x_offsets = {}, {}
    for off, _ in pairs(y_offsets_map) do table.insert(y_offsets, off) end
    for off, _ in pairs(x_offsets_map) do table.insert(x_offsets, off) end

    if #x_offsets > 0 and #y_offsets > 0 then
      local ok = verify_xy_candidates(region, x_offsets, y_offsets)
      if ok then
        CALIB.player_x_addr = region.start_addr + x_offsets[1]
        CALIB.player_y_addr = region.start_addr + y_offsets[1]
        CALIB.addr_domain = region.name
        save_calib()
        log_line(string.format('Calibration success: X=0x%08X Y=0x%08X in %s', CALIB.player_x_addr, CALIB.player_y_addr, region.name))
        return true
      end
    end
  end
  log_line('Calibration failed')
  return false
end

local function get_player_xy()
  if not CALIB.player_x_addr or not CALIB.player_y_addr then return nil end
  local x = read_u16_le(CALIB.player_x_addr)
  local y = read_u16_le(CALIB.player_y_addr)
  return x, y
end

-------------------------------------------------------------------------------
-- State Representation and Novelty Map
-------------------------------------------------------------------------------
local NOVEL = {
  -- key: cell_hash, value: visit_count
  visits = {},
  keys = {}, -- ring buffer of keys for capacity control
}

local PROGRESS = {
  unique_cells_total = 0,
  last_novelty_frame = 0,
  last_warp_frame = 0,
}

local function coarse_cell(x, y)
  if not x or not y then return 'unknown:unknown' end
  local cs = CONFIG.coarse_cell_size
  local cx = math.floor(x / cs)
  local cy = math.floor(y / cs)
  return string.format('%d:%d', cx, cy)
end

local function novelty_reward_for_cell(cell)
  local v = NOVEL.visits[cell] or 0
  NOVEL.visits[cell] = v + 1
  table.insert(NOVEL.keys, cell)
  if #NOVEL.keys > CONFIG.novelty_map_capacity then
    local old = table.remove(NOVEL.keys, 1)
    local ov = NOVEL.visits[old] or 1
    if ov <= 1 then NOVEL.visits[old] = nil end
  end
  if v == 0 then
    PROGRESS.unique_cells_total = PROGRESS.unique_cells_total + 1
    PROGRESS.last_novelty_frame = now_frame()
    return CONFIG.novelty_reward
  end
  return 0.0
end

-- Phase-aware state hash
local PHASE -- forward declare for hash_state
local function hash_state(x, y)
  if not x or not y then return 'noXY' end
  local cell = coarse_cell(x, y)
  local phase = (PHASE and PHASE.mode) or 'unknown'
  local age_bucket = math.floor(((PHASE and PHASE.frames_in_mode) or 0) / 60)
  return string.format('%s|%s|%d', cell, phase, age_bucket)
end

-------------------------------------------------------------------------------
-- Phase Detection and Dynamic Actions
-------------------------------------------------------------------------------
PHASE = {
  mode = 'overworld',  -- 'overworld' or 'interaction'
  stagnant_steps = 0,
  frames_in_mode = 0,
  last_x = nil,
  last_y = nil,
}

local function update_phase(x, y)
  local prev_mode = PHASE.mode
  if x and y and PHASE.last_x and PHASE.last_y then
    if x == PHASE.last_x and y == PHASE.last_y then
      PHASE.stagnant_steps = PHASE.stagnant_steps + 1
    else
      PHASE.stagnant_steps = 0
    end
  end
  PHASE.last_x, PHASE.last_y = x, y

  if PHASE.stagnant_steps >= CONFIG.phase_stagnant_threshold_steps then
    PHASE.mode = 'interaction'
  else
    PHASE.mode = 'overworld'
  end

  if PHASE.mode == prev_mode then
    PHASE.frames_in_mode = PHASE.frames_in_mode + 1
  else
    PHASE.frames_in_mode = 0
  end
  return prev_mode, PHASE.mode
end

-- Define action implementations (kept small and generic)
local ACTION_DEFS = {
  -- Overworld movement
  { name = 'Up',      phase = 'overworld', exec = function() tap('Up', CONFIG.frames_per_action) end },
  { name = 'Down',    phase = 'overworld', exec = function() tap('Down', CONFIG.frames_per_action) end },
  { name = 'Left',    phase = 'overworld', exec = function() tap('Left', CONFIG.frames_per_action) end },
  { name = 'Right',   phase = 'overworld', exec = function() tap('Right', CONFIG.frames_per_action) end },
  { name = 'A',       phase = 'any',       exec = function() tap('A', CONFIG.frames_per_action) end },
  { name = 'B',       phase = 'any',       exec = function() tap('B', CONFIG.frames_per_action) end },

  -- Interaction macros (battle/menu navigation heuristics)
  { name = 'MenuRightA', phase = 'interaction', exec = function() run_macro({ 'Right', 'A' }) end },
  { name = 'MenuDownA',  phase = 'interaction', exec = function() run_macro({ 'Down', 'A' }) end },
  { name = 'MenuLeftA',  phase = 'interaction', exec = function() run_macro({ 'Left', 'A' }) end },
  { name = 'MenuUpA',    phase = 'interaction', exec = function() run_macro({ 'Up', 'A' }) end },
  -- Run from battle (Fight/Bag/Pokemon/Run: bottom-right)
  { name = 'RunFromBattle', phase = 'interaction', exec = function() run_macro({ 'Down', 'Right', 'A' }) end },
  -- Attempt to throw a Poké Ball quickly: Right->Bag, A, Down (to Balls), A, A (use first ball)
  { name = 'BagThrowBallQuick', phase = 'interaction', exec = function() run_macro({ 'Right', 'A', 'Down', 'A', 'A' }) end },
  -- Attempt to fight using first move (A to Fight, A to first move)
  { name = 'FightFirst', phase = 'interaction', exec = function() run_macro({ 'A', 'A' }) end },
}

local function get_available_actions_for_phase(phase)
  local list = {}
  for _, def in ipairs(ACTION_DEFS) do
    if def.phase == 'any' or def.phase == phase then table.insert(list, def.name) end
  end
  return list
end

local function exec_action_by_name(name)
  for _, def in ipairs(ACTION_DEFS) do
    if def.name == name then def.exec(); return end
  end
  -- Fallback if unknown: press as single button
  tap(name)
end

-------------------------------------------------------------------------------
-- Feature Detection: Catch counter and optional map/badges
-------------------------------------------------------------------------------
local DETECT = {
  catch = { candidates = {}, best_addr = nil, prev_value = nil },
  map = { best_addr = nil, prev_value = nil },
  badges = { best_addr = nil, prev_value = nil },
}

local function record_catch_candidate(abs_addr)
  local s = DETECT.catch.candidates[abs_addr] or 0
  DETECT.catch.candidates[abs_addr] = s + 1
  -- Promote best if high confidence
  local best = DETECT.catch.best_addr
  if not best or DETECT.catch.candidates[abs_addr] > (DETECT.catch.candidates[best] or -1) then
    DETECT.catch.best_addr = abs_addr
    CALIB.caught_counter_addr = abs_addr
    save_calib()
  end
end

local function try_load_hints()
  local h = CONFIG.hints
  if h then
    if h.pokedex_caught_addr then CALIB.caught_counter_addr = parse_hex_or_num(h.pokedex_caught_addr) end
    if h.map_id_addr then CALIB.map_id_addr = parse_hex_or_num(h.map_id_addr) end
    if h.badge_flags_addr then CALIB.badge_flags_addr = parse_hex_or_num(h.badge_flags_addr) end
  end
end

local function read_optional_u16(addr)
  if not addr then return nil end
  local ok, val = pcall(read_u16_le, addr)
  if ok then return val end
  return nil
end

local function on_possible_catch_event(before_snaps, after_snaps, regions)
  -- Scan for +1 in 16-bit counters across regions; record candidates
  for _, region in ipairs(regions) do
    local before = before_snaps[region.name]
    local after = after_snaps[region.name]
    if before and after then
      local size = math.min(#before, #after)
      for i = 1, size - 1, 2 do
        local vb = before[i] + 256 * before[i + 1]
        local va = after[i] + 256 * after[i + 1]
        if va - vb == 1 then
          local abs_addr = region.start_addr + (i - 1)
          record_catch_candidate(abs_addr)
        end
      end
    end
  end
end

-------------------------------------------------------------------------------
-- Savestate Manager
-------------------------------------------------------------------------------
local SAVE = {
  enabled = false,
  strategy = nil, -- 'slot' or 'file'
  next_slot = 1,
  saved_slots = 0,
  last_saved_unique_cells = 0,
}

local function save_slot(idx)
  if not SAVE.enabled then return false end
  if SAVE.strategy == 'slot' and savestate and savestate.saveslot then
    local ok = pcall(savestate.saveslot, idx)
    return ok
  elseif SAVE.strategy == 'file' and savestate and savestate.save then
    local path = string.format('%s/emerald_ai_slot_%d.State', CONFIG.save_dir, idx)
    local ok = pcall(savestate.save, path)
    return ok
  end
  return false
end

local function load_slot(idx)
  if not SAVE.enabled then return false end
  if SAVE.strategy == 'slot' and savestate and savestate.loadslot then
    local ok = pcall(savestate.loadslot, idx)
    return ok
  elseif SAVE.strategy == 'file' and savestate and savestate.load then
    local path = string.format('%s/emerald_ai_slot_%d.State', CONFIG.save_dir, idx)
    local ok = pcall(savestate.load, path)
    return ok
  end
  return false
end

local function maybe_save_progress()
  if not SAVE.enabled then return end
  if PROGRESS.unique_cells_total - SAVE.last_saved_unique_cells >= CONFIG.savestate_save_every_new_cells then
    local idx = SAVE.next_slot
    if save_slot(idx) then
      SAVE.saved_slots = math.max(SAVE.saved_slots, idx)
      SAVE.next_slot = (idx % CONFIG.savestate_slots) + 1
      SAVE.last_saved_unique_cells = PROGRESS.unique_cells_total
      log_line(string.format('Savestate saved to slot %d at unique_cells=%d', idx, PROGRESS.unique_cells_total))
    end
  end
end

local function maybe_reset_no_progress()
  if not SAVE.enabled then return end
  local frame = now_frame()
  if (frame - PROGRESS.last_novelty_frame) >= CONFIG.no_novelty_reset_frames and SAVE.saved_slots > 0 then
    local idx = math.random(1, math.min(SAVE.saved_slots, CONFIG.savestate_slots))
    if load_slot(idx) then
      PROGRESS.last_novelty_frame = now_frame()
      log_line(string.format('Savestate loaded from slot %d due to no novelty', idx))
    end
  end
end

local function detect_savestate_capabilities()
  if not CONFIG.savestate_enable or not savestate then return end
  if savestate.saveslot and savestate.loadslot then
    SAVE.strategy = 'slot'
    SAVE.enabled = true
  elseif savestate.save and savestate.load then
    SAVE.strategy = 'file'
    SAVE.enabled = true
  else
    SAVE.enabled = false
  end
end

-------------------------------------------------------------------------------
-- Actions and Environment Step
-------------------------------------------------------------------------------
local ACTIONS = { 'Up', 'Down', 'Left', 'Right', 'A', 'B' }

local function choose_action(state_hash, available_actions, epsilon)
  if math.random() < epsilon then
    return available_actions[math.random(1, #available_actions)]
  end
  -- Greedy
  local best_a, best_q = available_actions[1], -1e9
  for _, a in ipairs(available_actions) do
    local entry = QTABLE[q_key(state_hash, a)]
    local q = entry and entry.q or 0.0
    if q > best_q then best_q = q; best_a = a end
  end
  return best_a
end

local function update_q(state_hash, action, reward, next_state_hash)
  local key = q_key(state_hash, action)
  local entry = QTABLE[key] or { q = 0.0, n = 0 }
  -- Max over actions in next state (consider all defined actions)
  local max_next = -1e9
  for _, def in ipairs(ACTION_DEFS) do
    local a = def.name
    local e = QTABLE[q_key(next_state_hash, a)]
    local q = e and e.q or 0.0
    if q > max_next then max_next = q end
  end
  if max_next == -1e9 then max_next = 0.0 end

  local lr = CONFIG.learning_rate
  local gamma = CONFIG.discount_factor
  entry.q = entry.q + lr * (reward + gamma * max_next - entry.q)
  entry.n = entry.n + 1
  QTABLE[key] = entry
end

local function step_action(action)
  local x0, y0 = get_player_xy()
  exec_action_by_name(action)
  local x1, y1 = get_player_xy()

  local reward = 0.0
  if x0 and y0 and x1 and y1 then
    local moved = (x0 ~= x1) or (y0 ~= y1)
    if moved then reward = reward + CONFIG.small_move_reward end
    -- Warp detection by large XY jump
    local manhattan = (x0 and x1) and (math.abs(x1 - x0) + math.abs(y1 - y0)) or 0
    if manhattan >= CONFIG.warp_distance_threshold then
      reward = reward + CONFIG.warp_reward
      PROGRESS.last_warp_frame = now_frame()
    end
  end

  -- Novelty based on coarse cell
  reward = reward + novelty_reward_for_cell(coarse_cell(x1, y1))

  return { x = x1, y = y1, reward = reward }
end

-------------------------------------------------------------------------------
-- Epsilon Schedule
-------------------------------------------------------------------------------
local function epsilon_for_frame(frame)
  local e0, emin, decay = CONFIG.epsilon_start, CONFIG.epsilon_min, CONFIG.epsilon_decay_frames
  if frame >= decay then return emin end
  local t = frame / decay
  return e0 + (emin - e0) * t
end

-------------------------------------------------------------------------------
-- Main Loop
-------------------------------------------------------------------------------
local function ensure_calibrated()
  load_calib()
  try_load_hints()
  if CALIB.player_x_addr and CALIB.player_y_addr then return true end
  return auto_calibrate_xy()
end

local RECENT = {
  catch_attempt_frame = nil,
  catch_snap_before = nil, -- table by region name to byte array
  last_xy = { x = nil, y = nil, stagnant = 0 },
}

local function init()
  math.randomseed(os.time() % 2147483647)
  load_meta()
  load_qtable()
  ensure_calibrated()
  detect_savestate_capabilities()
  if SAVE.enabled then
    save_slot(1)
    SAVE.saved_slots = 1
    SAVE.next_slot = 2
  end
  -- Initialize feature readers
  if CALIB.caught_counter_addr then
    DETECT.catch.best_addr = CALIB.caught_counter_addr
    DETECT.catch.prev_value = read_optional_u16(CALIB.caught_counter_addr)
  end
  if CALIB.map_id_addr then
    DETECT.map.best_addr = CALIB.map_id_addr
    DETECT.map.prev_value = read_optional_u16(CALIB.map_id_addr)
  end
  if CALIB.badge_flags_addr then
    DETECT.badges.best_addr = CALIB.badge_flags_addr
    DETECT.badges.prev_value = read_optional_u16(CALIB.badge_flags_addr)
  end
  log_line('Init done; emulator=' .. EMU.name)
end

local function main_loop()
  local last_save_frame = now_frame()
  local steps = 0
  while true do
    local frame = now_frame()
    if (frame - last_save_frame) >= CONFIG.autosave_interval_frames then
      save_qtable(); save_meta(); save_calib()
      last_save_frame = frame
      log_line('Autosaved state')
    end

    local x, y = get_player_xy()
    if not x or not y then
      -- Try to recalibrate occasionally
      if frame % 2000 == 0 then ensure_calibrated() end
      -- Fallback: random input to progress
      exec_action_by_name(ACTIONS[math.random(1, #ACTIONS)])
      goto continue
    end

    -- Update phase, choose actions accordingly
    update_phase(x, y)
    local available_actions = get_available_actions_for_phase(PHASE.mode)
    local state_hash = hash_state(x, y)
    local eps = epsilon_for_frame(frame)

    -- Choose action; if throwing a ball, record pre-snapshots for detection
    local action = choose_action(state_hash, available_actions, eps)
    local pre_catch_before = nil
    if action == 'BagThrowBallQuick' then
      pre_catch_before = {}
      for _, region in ipairs(CONFIG.calib_scan_regions) do
        pre_catch_before[region.name] = snapshot_region(region)
      end
      RECENT.catch_attempt_frame = frame
      RECENT.catch_snap_before = pre_catch_before
    end

    local outcome = step_action(action)

    -- Phase transition shaping and catch detection window
    local before_mode = PHASE.mode
    local _, new_mode = update_phase(outcome.x, outcome.y)
    if before_mode == 'interaction' and new_mode == 'overworld' then
      outcome.reward = outcome.reward + CONFIG.interaction_exit_reward
      -- If a catch attempt was recent, scan for +1 counters
      if RECENT.catch_attempt_frame and (frame - RECENT.catch_attempt_frame) <= CONFIG.catch_attempt_window_frames and RECENT.catch_snap_before then
        local after_snaps = {}
        for _, region in ipairs(CONFIG.calib_scan_regions) do
          after_snaps[region.name] = snapshot_region(region)
        end
        on_possible_catch_event(RECENT.catch_snap_before, after_snaps, CONFIG.calib_scan_regions)
        RECENT.catch_attempt_frame = nil
        RECENT.catch_snap_before = nil
      end
    elseif new_mode == 'interaction' and (PHASE.frames_in_mode % 60 == 0) then
      outcome.reward = outcome.reward + CONFIG.long_interaction_penalty
    end

    -- If we have a reliable caught counter, reward on increments
    if DETECT.catch.best_addr then
      local cur = read_optional_u16(DETECT.catch.best_addr)
      if cur and DETECT.catch.prev_value and cur > DETECT.catch.prev_value then
        outcome.reward = outcome.reward + CONFIG.catch_reward
      end
      DETECT.catch.prev_value = cur or DETECT.catch.prev_value
    end

    -- If we have a map id address, reward on new map id transitions
    if DETECT.map.best_addr then
      local cur = read_optional_u16(DETECT.map.best_addr)
      if cur and DETECT.map.prev_value and cur ~= DETECT.map.prev_value then
        outcome.reward = outcome.reward + CONFIG.new_map_reward
      end
      DETECT.map.prev_value = cur or DETECT.map.prev_value
    end

    -- If we have badge flags address, reward on new bits set
    if DETECT.badges.best_addr then
      local cur = read_optional_u16(DETECT.badges.best_addr)
      if cur and DETECT.badges.prev_value then
        local gained = (cur | 0) & (~(DETECT.badges.prev_value | 0))
        if gained ~= 0 then outcome.reward = outcome.reward + CONFIG.badge_reward end
      end
      DETECT.badges.prev_value = cur or DETECT.badges.prev_value
    end

    local next_state_hash = hash_state(outcome.x, outcome.y)
    update_q(state_hash, action, outcome.reward, next_state_hash)

    META.total_steps = (META.total_steps or 0) + 1
    steps = steps + 1

    -- Stuck detection: if coarse position unchanged for long, penalize and randomize
    if RECENT.last_xy.x and RECENT.last_xy.y and outcome.x == RECENT.last_xy.x and outcome.y == RECENT.last_xy.y then
      RECENT.last_xy.stagnant = RECENT.last_xy.stagnant + 1
    else
      RECENT.last_xy.stagnant = 0
    end
    RECENT.last_xy.x, RECENT.last_xy.y = outcome.x, outcome.y
    if RECENT.last_xy.stagnant > 120 then
      local s = hash_state(outcome.x, outcome.y)
      update_q(s, action, CONFIG.stuck_penalty, s)
      -- Try to break out: mash A and random direction
      exec_action_by_name('A')
      exec_action_by_name(ACTIONS[math.random(1, 4)])
      RECENT.last_xy.stagnant = 0
    end

    -- Savestates: save on progress; reset on long stagnation
    maybe_save_progress()
    maybe_reset_no_progress()

    if steps % CONFIG.log_every_n_steps == 0 then
      log_line(string.format('step=%d eps=%.3f action=%s reward=%.3f pos=(%s) phase=%s(%d) unique=%d', steps, eps, action, outcome.reward, next_state_hash, PHASE.mode, PHASE.frames_in_mode, PROGRESS.unique_cells_total))
    end

    ::continue::
    emu.frameadvance()
  end
end

-------------------------------------------------------------------------------
-- Entry
-------------------------------------------------------------------------------
init()
main_loop()
