-- emerald_ai.lua
-- Single-file self-learning agent for Pokémon Emerald (GBA)
-- Target: BizHawk or mGBA Lua environments (emulator-agnostic wrappers)
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
  journal_filename = 'emerald_ai_playthrough.md',

  -- RL parameters
  epsilon_start = 0.20,
  epsilon_min = 0.02,
  epsilon_decay_frames = 2e6,
  learning_rate = 0.20,
  discount_factor = 0.995,

  -- Control loop
  frames_per_action = 10,
  frames_between_steps = 2,
  autosave_interval_frames = 2000,

  -- Novelty / reward
  novelty_reward = 0.5,
  stuck_penalty = -0.5,
  small_move_reward = 0.05,

  -- Phase/interaction shaping
  phase_stagnant_threshold_steps = 20,
  interaction_exit_reward = 1.0,
  long_interaction_penalty = -0.2,
  max_interaction_steps_without_exit = 200,
  macro_hold_frames = 6,

  -- Higher-level shaping
  catch_attempt_window_frames = 1200,
  catch_reward = 5.0,
  warp_reward = 1.0,
  new_map_reward = 1.0,
  badge_reward = 20.0,
  frontier_entry_reward = 2.0,
  warp_distance_threshold = 24,

  -- Ball usage heuristics
  ball_throw_cooldown_frames = 1200,
  ball_low_threshold = 5,

  -- Healing heuristics
  heal_min_ratio = 0.40,           -- if min party HP ratio < this, prefer CenterHeal
  heal_check_interval_frames = 900, -- ~15s

  -- Episode management
  episode_badge_mask = 0x00FF,
  lower_epsilon_per_episode = 0.95,

  -- Savestates / episodes
  savestate_enable = true,
  savestate_slots = 5,
  savestate_save_every_new_cells = 100,
  no_novelty_reset_frames = 6000,

  -- Screenshots (BizHawk only)
  screenshots_on_events = true,

  -- Coordinate detection
  calib_move_frames = 18,
  calib_scan_regions = {
    {name = 'EWRAM', start_addr = 0x02000000, size = 0x40000},
    {name = 'IWRAM', start_addr = 0x03000000, size = 0x08000},
  },
  calib_try_directions = {'Up', 'Down', 'Left', 'Right'},
  calib_candidate_max = 64,

  -- Coarse grid for novelty map
  coarse_cell_size = 4,
  novelty_map_capacity = 100000,

  -- Optional manual memory hints (numbers or hex strings; lists as comma strings or arrays)
  hints = {
    pokedex_caught_addr = nil, -- u16
    map_id_addr = nil,         -- u16
    badge_flags_addr = nil,    -- u16
    battle_flag_addr = nil,    -- u8/u16 (non-zero when in battle)
    bag_pokeball_count_addr = nil, -- u16
    party_hp_current_addrs = nil,   -- list
    party_hp_max_addrs = nil,       -- list
  },

  -- Map awareness (optional) - users can fill these
  map_sets = {
    center_map_ids = {},
    mart_map_ids = {},
    frontier_map_ids = {},
  },
  map_names = {},

  -- Logging verbosity
  log_every_n_steps = 200,
}

-------------------------------------------------------------------------------
-- Emulator Abstraction Layer
-------------------------------------------------------------------------------
local EMU = { name = 'unknown', has_memory_domains = false, memory_domain = nil, joypad_port = 1 }

local function detect_emulator()
  if client and type(client.getversion) == 'function' then
    EMU.name = 'bizhawk'
  elseif emu and type(emu.framecount) == 'function' and package and package.cpath and not client then
    EMU.name = 'mgba'
  else
    EMU.name = 'unknown'
  end
  if memory and type(memory.usememorydomain) == 'function' then EMU.has_memory_domains = true end
  if EMU.has_memory_domains then
    local ok = pcall(function() memory.usememorydomain('System Bus') end)
    if ok then EMU.memory_domain = 'System Bus' end
  end
end

detect_emulator()

-------------------------------------------------------------------------------
-- Utilities: Logging, Files, Journal, Screenshots
-------------------------------------------------------------------------------
local function now_frame() if emu and emu.framecount then return emu.framecount() end return 0 end

local function write_file(path, text) local f = io.open(path, 'w'); if not f then return false end f:write(text) f:close() return true end
local function append_file(path, text) local f = io.open(path, 'a'); if not f then return false end f:write(text) f:close() return true end
local function read_file(path) local f = io.open(path, 'r'); if not f then return nil end local c=f:read('*a'); f:close(); return c end

local function log_line(msg)
  local line = string.format('[%d][%s] %s\n', now_frame(), EMU.name, msg)
  append_file(CONFIG.save_dir .. '/' .. CONFIG.log_filename, line)
end

local function frames_to_hms(f)
  local fps = 60; local sec = math.floor(f / fps)
  local h = math.floor(sec / 3600); local m = math.floor((sec % 3600) / 60); local s = sec % 60
  return string.format('%02d:%02d:%02d', h, m, s)
end

local function journal_append(text) append_file(CONFIG.save_dir .. '/' .. CONFIG.journal_filename, text) end
local function journal_event(title, body)
  local line = string.format('- %s at %s (F%d): %s\n', title, frames_to_hms(now_frame()), now_frame(), body or '')
  journal_append(line)
end

local function take_screenshot(tag)
  if not CONFIG.screenshots_on_events then return end
  if EMU.name == 'bizhawk' and client and client.screenshot then
    local path = string.format('%s/emerald_ai_shot_E%d_F%d_%s.png', CONFIG.save_dir, (META.episode or 0), now_frame(), tostring(tag or 'event'))
    pcall(client.screenshot, path)
  end
end

local function map_name(id)
  if not id then return 'Unknown' end
  local n = (CONFIG.map_names or {})[id]; if n then return n end
  return string.format('Map %d', id)
end

-------------------------------------------------------------------------------
-- Bitwise helpers
-------------------------------------------------------------------------------
local function bit_and(a, b)
  if bit and bit.band then return bit.band(a, b) end
  if bit32 and bit32.band then return bit32.band(a, b) end
  local ok, res = pcall(function() return a & b end); if ok then return res end
  local res2, bitv = 0, 1; while a > 0 or b > 0 do local abit=a%2; local bbit=b%2; if abit==1 and bbit==1 then res2=res2+bitv end; a=math.floor(a/2); b=math.floor(b/2); bitv=bitv*2 end; return res2
end
local function bit_or(a, b)
  if bit and bit.bor then return bit.bor(a, b) end
  if bit32 and bit32.bor then return bit32.bor(a, b) end
  local ok, res = pcall(function() return a | b end); if ok then return res end
  local res2, bitv = 0, 1; while a > 0 or b > 0 do local abit=a%2; local bbit=b%2; if abit==1 or bbit==1 then res2=res2+bitv end; a=math.floor(a/2); b=math.floor(b/2); bitv=bitv*2 end; return res2
end
local function bit_not16(x) return 0xFFFF - (x % 0x10000) end

-------------------------------------------------------------------------------
-- Joypad
-------------------------------------------------------------------------------
local function joypad_set(buttons_table)
  if joypad and type(joypad.set) == 'function' then
    local ok = pcall(function()
      if EMU.name == 'bizhawk' then joypad.set(EMU.joypad_port, buttons_table) else joypad.set(buttons_table) end
    end); if ok then return end
  end
end

local function press_buttons(button_names, hold_frames)
  hold_frames = hold_frames or CONFIG.frames_per_action
  local buttons = {}; for _, name in ipairs(button_names) do buttons[name] = true end
  joypad_set(buttons); for _=1,hold_frames do emu.frameadvance() end
  joypad_set({}); for _=1,CONFIG.frames_between_steps do emu.frameadvance() end
end
local function press_button(name, hold_frames) press_buttons({name}, hold_frames) end
local function tap(name, hold_frames) press_button(name, hold_frames or CONFIG.macro_hold_frames) end
local function run_macro(seq) for _, step in ipairs(seq) do if type(step)=='table' then tap(step[1], step[2]) else tap(step) end end end

-------------------------------------------------------------------------------
-- Memory wrappers
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
  local lo = read_u8(addr); local hi = read_u8(addr+1); return lo + hi*256
end
local function read_bytes(addr, count)
  if memory and memory.read_bytes_as_array then return memory.read_bytes_as_array(addr, count) end
  local arr = {}; for i=0,count-1 do arr[i+1]=read_u8(addr+i) end; return arr
end

-------------------------------------------------------------------------------
-- Persistence (meta, Q-table)
-------------------------------------------------------------------------------
local META = { total_steps = 0, total_episodes = 0, episode = 0 }
local function load_meta() local p=CONFIG.save_dir..'/'..CONFIG.meta_filename; local c=read_file(p); if not c then return end; for line in c:gmatch('[^\n]+') do local k,v=line:match('^(%w+)\t(.+)$'); if k and v then if tonumber(v) then META[k]=tonumber(v) else META[k]=v end end end end
local function save_meta() local p=CONFIG.save_dir..'/'..CONFIG.meta_filename; local lines={}; for k,v in pairs(META) do table.insert(lines, string.format('%s\t%s', k, tostring(v))) end; write_file(p, table.concat(lines,'\n')) end

local QTABLE = {}
local function q_key(state_hash, action_name) return state_hash..'\t'..action_name end
local function load_qtable() local p=CONFIG.save_dir..'/'..CONFIG.qtable_filename; local c=read_file(p); if not c then return end; for line in c:gmatch('[^\n]+') do local s,a,qs,ns=line:match('^(.+)\t([^\t]+)\t([^\t]+)\t([^\t]+)$'); if s and a and qs and ns then QTABLE[q_key(s,a)]={q=tonumber(qs) or 0.0, n=tonumber(ns) or 0} end end end
local function save_qtable() local p=CONFIG.save_dir..'/'..CONFIG.qtable_filename; local lines={}; for key,e in pairs(QTABLE) do local s,a=key:match('^(.*)\t([^\t]+)$'); table.insert(lines, string.format('%s\t%s\t%.6f\t%d', s or '', a or '', e.q or 0.0, e.n or 0)) end; write_file(p, table.concat(lines,'\n')) end

-------------------------------------------------------------------------------
-- Calibration + Feature detection persistence
-------------------------------------------------------------------------------
local CALIB = {
  player_x_addr = nil,
  player_y_addr = nil,
  addr_domain = 'auto',
  caught_counter_addr = nil,
  map_id_addr = nil,
  badge_flags_addr = nil,
}

local function save_calib()
  local p=CONFIG.save_dir..'/'..CONFIG.calib_filename; local lines={}
  for k,v in pairs(CALIB) do if type(v)=='number' then table.insert(lines, string.format('%s\t0x%08X', k, v)) else table.insert(lines, string.format('%s\t%s', k, tostring(v))) end end
  write_file(p, table.concat(lines,'\n'))
end

local function parse_hex_or_num(s) if not s or s=='nil' then return nil end; if type(s)=='number' then return s end; local hex=s:match('^0x([0-9a-fA-F]+)$'); if hex then return tonumber(hex,16) end; return tonumber(s) end

local function split_comma_values(val)
  if not val then return {} end
  if type(val) == 'table' then return val end
  local t = {}
  if type(val) == 'string' then
    for part in val:gmatch('[^,]+') do t[#t+1] = part:gsub('^%s+',''):gsub('%s+$','') end
  end
  return t
end

local DETECT = {
  catch = { candidates = {}, best_addr = nil, prev_value = nil },
  map   = { best_addr = nil, prev_value = nil },
  badges= { best_addr = nil, prev_value = nil },
  battle= { best_addr = nil, prev_value = nil },
  hp    = { current_addrs = {}, max_addrs = {} },
  bag   = { ball_count_addr = nil, prev_count = nil },
}

local function load_calib()
  local p=CONFIG.save_dir..'/'..CONFIG.calib_filename; local c=read_file(p); if not c then return end
  for line in c:gmatch('[^\n]+') do local k,v=line:match('^(%w+)\t(.+)$'); if k and v then local num=parse_hex_or_num(v); if num~=nil then CALIB[k]=num else CALIB[k]=v end end end
end

local function try_load_hints()
  local h = CONFIG.hints or {}
  if h.pokedex_caught_addr then CALIB.caught_counter_addr = parse_hex_or_num(h.pokedex_caught_addr) end
  if h.map_id_addr then CALIB.map_id_addr = parse_hex_or_num(h.map_id_addr) end
  if h.badge_flags_addr then CALIB.badge_flags_addr = parse_hex_or_num(h.badge_flags_addr) end
  if h.battle_flag_addr then DETECT.battle.best_addr = parse_hex_or_num(h.battle_flag_addr) end
  if h.bag_pokeball_count_addr then DETECT.bag.ball_count_addr = parse_hex_or_num(h.bag_pokeball_count_addr) end
  local curr_list = split_comma_values(h.party_hp_current_addrs)
  local max_list  = split_comma_values(h.party_hp_max_addrs)
  DETECT.hp.current_addrs = {}
  for _, it in ipairs(curr_list) do local n=parse_hex_or_num(it); if n then table.insert(DETECT.hp.current_addrs, n) end end
  DETECT.hp.max_addrs = {}
  for _, it in ipairs(max_list) do local n=parse_hex_or_num(it); if n then table.insert(DETECT.hp.max_addrs, n) end end
end

-------------------------------------------------------------------------------
-- Region snapshots & XY calibration
-------------------------------------------------------------------------------
local function snapshot_region(region) return read_bytes(region.start_addr, region.size) end
local function diff_candidates(before, after)
  local candidates = {}; local limit=CONFIG.calib_candidate_max; local size=math.min(#before,#after)
  for i=1,size-1,2 do local lo_b,hi_b=before[i],before[i+1]; local lo_a,hi_a=after[i],after[i+1]; if lo_b and hi_b and lo_a and hi_a then local vb=lo_b+256*hi_b; local va=lo_a+256*hi_a; local d=va-vb; if d~=0 and math.abs(d)<=8 then table.insert(candidates,{offset=i-1,delta=d}); if #candidates>=limit then break end end end end
  return candidates
end
local function try_find_xy_in_region(region, dir) local base=snapshot_region(region); press_button(dir, CONFIG.calib_move_frames); local moved=snapshot_region(region); local reverse=({Up='Down',Down='Up',Left='Right',Right='Left'})[dir]; if reverse then press_button(reverse, CONFIG.calib_move_frames) end; return diff_candidates(base,moved) end
local function verify_xy_candidates(region, x_offsets, y_offsets)
  local base=snapshot_region(region); press_button('Right', CONFIG.calib_move_frames); local after_right=snapshot_region(region); press_button('Left', CONFIG.calib_move_frames)
  local function val_at(arr, off) local lo=arr[off+1]; local hi=arr[off+2]; return lo+256*hi end
  local function avg_delta(offsets, a, b) if #offsets==0 then return 0 end; local sum,n=0,0; for _,off in ipairs(offsets) do sum=sum+(val_at(a,off)-val_at(b,off)); n=n+1 end; return sum/math.max(n,1) end
  local dx=avg_delta(x_offsets,after_right,base); local dy=avg_delta(y_offsets,after_right,base); return (math.abs(dx)>=1 and math.abs(dy)<=1)
end
local function auto_calibrate_xy()
  log_line('Calibration started')
  for _,region in ipairs(CONFIG.calib_scan_regions) do
    local dir_cands={}; for _,dir in ipairs(CONFIG.calib_try_directions) do dir_cands[dir]=try_find_xy_in_region(region,dir) end
    local ymap,xmap={},{}; for _,d in ipairs({'Up','Down'}) do for _,c in ipairs(dir_cands[d] or {}) do ymap[c.offset]=(ymap[c.offset] or 0)+1 end end; for _,d in ipairs({'Left','Right'}) do for _,c in ipairs(dir_cands[d] or {}) do xmap[c.offset]=(xmap[c.offset] or 0)+1 end end
    local yoffs,xoffs={},{}; for off,_ in pairs(ymap) do table.insert(yoffs,off) end; for off,_ in pairs(xmap) do table.insert(xoffs,off) end
    if #xoffs>0 and #yoffs>0 then if verify_xy_candidates(region,xoffs,yoffs) then CALIB.player_x_addr=region.start_addr+xoffs[1]; CALIB.player_y_addr=region.start_addr+yoffs[1]; CALIB.addr_domain=region.name; save_calib(); log_line(string.format('Calibration success: X=0x%08X Y=0x%08X in %s', CALIB.player_x_addr, CALIB.player_y_addr, region.name)); return true end end
  end
  log_line('Calibration failed'); return false
end
local function get_player_xy() if not CALIB.player_x_addr or not CALIB.player_y_addr then return nil end; return read_u16_le(CALIB.player_x_addr), read_u16_le(CALIB.player_y_addr) end

-------------------------------------------------------------------------------
-- State & Novelty
-------------------------------------------------------------------------------
local NOVEL = { visits = {}, keys = {} }
local PROGRESS = { unique_cells_total = 0, last_novelty_frame = 0, last_warp_frame = 0 }
local PHASE -- forward declaration

local function coarse_cell(x, y) if not x or not y then return 'unknown:unknown' end; local cs=CONFIG.coarse_cell_size; return string.format('%d:%d', math.floor(x/cs), math.floor(y/cs)) end
local function novelty_reward_for_cell(cell)
  local v = NOVEL.visits[cell] or 0; NOVEL.visits[cell]=v+1; table.insert(NOVEL.keys, cell)
  if #NOVEL.keys>CONFIG.novelty_map_capacity then local old=table.remove(NOVEL.keys,1); local ov=NOVEL.visits[old] or 1; if ov<=1 then NOVEL.visits[old]=nil end end
  if v==0 then PROGRESS.unique_cells_total=PROGRESS.unique_cells_total+1; PROGRESS.last_novelty_frame=now_frame(); return CONFIG.novelty_reward end
  return 0.0
end
local function hash_state(x,y) if not x or not y then return 'noXY' end; local cell=coarse_cell(x,y); local phase=(PHASE and PHASE.mode) or 'unknown'; local age_bucket=math.floor(((PHASE and PHASE.frames_in_mode) or 0)/60); return string.format('%s|%s|%d', cell, phase, age_bucket) end

-------------------------------------------------------------------------------
-- Phase detection & Dynamic actions
-------------------------------------------------------------------------------
PHASE = { mode = 'overworld', stagnant_steps = 0, frames_in_mode = 0, last_x = nil, last_y = nil }
local function update_phase(x,y)
  local prev=PHASE.mode; if x and y and PHASE.last_x and PHASE.last_y then if x==PHASE.last_x and y==PHASE.last_y then PHASE.stagnant_steps=PHASE.stagnant_steps+1 else PHASE.stagnant_steps=0 end end; PHASE.last_x,PHASE.last_y=x,y
  if PHASE.stagnant_steps>=CONFIG.phase_stagnant_threshold_steps then PHASE.mode='interaction' else PHASE.mode='overworld' end
  if PHASE.mode==prev then PHASE.frames_in_mode=PHASE.frames_in_mode+1 else PHASE.frames_in_mode=0 end
  return prev, PHASE.mode
end

-- Utility: map context
local function get_map_context()
  local id = DETECT.map.prev_value
  local sets = CONFIG.map_sets or {}
  local center = id and ((sets.center_map_ids or {})[id] == true) or false
  local mart   = id and ((sets.mart_map_ids or {})[id] == true) or false
  local frontier = id and ((sets.frontier_map_ids or {})[id] == true) or false
  return id, center, mart, frontier
end

-- Heuristics: battle flag, party health, and balls
local function in_battle()
  if DETECT.battle.best_addr then
    local v = read_u8(DETECT.battle.best_addr)
    if v and v ~= 0 then return true end
  end
  return PHASE.mode == 'interaction' and PHASE.frames_in_mode > 15
end
local function compute_party_health_ratio()
  local currs, maxs = DETECT.hp.current_addrs or {}, DETECT.hp.max_addrs or {}
  local n = math.min(#currs, #maxs); if n == 0 then return nil, nil end
  local sum_ratio, min_ratio, cnt = 0.0, 1.0, 0
  for i=1,n do
    local c = read_u16_le(currs[i]) or 0
    local m = read_u16_le(maxs[i]) or 1
    if m <= 0 then m = 1 end
    local ratio = math.max(0.0, math.min(1.0, c / m))
    if ratio < min_ratio then min_ratio = ratio end
    sum_ratio = sum_ratio + ratio
    cnt = cnt + 1
  end
  if cnt == 0 then return nil, nil end
  return min_ratio, (sum_ratio / cnt)
end
local function needs_heal()
  local minr = select(1, compute_party_health_ratio())
  return (minr ~= nil) and (minr < CONFIG.heal_min_ratio)
end
local function get_ball_count()
  if DETECT.bag.ball_count_addr then return read_u16_le(DETECT.bag.ball_count_addr) end
  return nil
end
local function balls_low()
  local c = get_ball_count(); if c == nil then return false end; return c < CONFIG.ball_low_threshold
end

-- Actions
local ACTION_DEFS = {
  -- Overworld movement
  { name = 'Up',      phase = 'overworld',   exec = function() tap('Up', CONFIG.frames_per_action) end },
  { name = 'Down',    phase = 'overworld',   exec = function() tap('Down', CONFIG.frames_per_action) end },
  { name = 'Left',    phase = 'overworld',   exec = function() tap('Left', CONFIG.frames_per_action) end },
  { name = 'Right',   phase = 'overworld',   exec = function() tap('Right', CONFIG.frames_per_action) end },
  { name = 'A',       phase = 'any',         exec = function() tap('A', CONFIG.frames_per_action) end },
  { name = 'B',       phase = 'any',         exec = function() tap('B', CONFIG.frames_per_action) end },
  -- Interaction macros
  { name = 'MenuRightA',        phase = 'interaction', exec = function() run_macro({ 'Right', 'A' }) end },
  { name = 'MenuDownA',         phase = 'interaction', exec = function() run_macro({ 'Down', 'A' }) end },
  { name = 'MenuLeftA',         phase = 'interaction', exec = function() run_macro({ 'Left', 'A' }) end },
  { name = 'MenuUpA',           phase = 'interaction', exec = function() run_macro({ 'Up', 'A' }) end },
  { name = 'RunFromBattle',     phase = 'interaction', exec = function() run_macro({ 'Down', 'Right', 'A' }) end },
  { name = 'BagThrowBallQuick', phase = 'interaction', exec = function() run_macro({ 'Right', 'A', 'Down', 'A', 'A' }) end },
  { name = 'FightFirst',        phase = 'interaction', exec = function() run_macro({ 'A', 'A' }) end },
  { name = 'BattleSwitchNext',  phase = 'interaction', exec = function() run_macro({ 'Down', 'A', 'Down', 'A', 'A' }) end },
  -- Overworld helpers
  { name = 'CenterHeal',        phase = 'any',         exec = function() run_macro({ 'Up', 'A', 'A', 'A', 'A', 'A', 'B' }) end },
  { name = 'MartBuyBallsQuick', phase = 'any',         exec = function() run_macro({ 'Up', 'A', 'A', 'A', 'A', 'B', 'B' }) end },
  { name = 'MartBuyBalls10',    phase = 'any',         exec = function() run_macro({ 'A', 'Right', 'Right', 'A', 'A', 'B' }) end },
}

local function get_available_actions_for_phase(phase)
  local list = {}
  local id, inCenter, inMart = get_map_context()
  local healOk = needs_heal()
  local lowBalls = balls_low()
  local canThrow = (get_ball_count() == nil) or ((get_ball_count() or 0) > 0)
  for _, def in ipairs(ACTION_DEFS) do
    if def.phase == 'any' or def.phase == phase then
      local name = def.name; local allow = true
      if name == 'CenterHeal' then
        if not inCenter or not healOk then allow = false end
      elseif name == 'MartBuyBallsQuick' or name == 'MartBuyBalls10' then
        if not inMart or not lowBalls then allow = false end
      elseif name == 'BagThrowBallQuick' then
        if not canThrow then allow = false end
      end
      if allow then table.insert(list, name) end
    end
  end
  -- Ensure fallbacks exist
  if #list == 0 then list = { 'Up', 'Down', 'Left', 'Right', 'A', 'B' } end
  return list
end

local function exec_action_by_name(name) for _,def in ipairs(ACTION_DEFS) do if def.name==name then def.exec(); return end end tap(name) end

-------------------------------------------------------------------------------
-- Savestate Manager
-------------------------------------------------------------------------------
local SAVE = { enabled = false, strategy = nil, next_slot = 1, saved_slots = 0, last_saved_unique_cells = 0 }
local function save_slot(idx)
  if not SAVE.enabled then return false end
  if SAVE.strategy == 'slot' and savestate and savestate.saveslot then return pcall(savestate.saveslot, idx)
  elseif SAVE.strategy == 'file' and savestate and savestate.save then local path=string.format('%s/emerald_ai_slot_%d.State', CONFIG.save_dir, idx); return pcall(savestate.save, path) end
  return false
end
local function load_slot(idx)
  if not SAVE.enabled then return false end
  if SAVE.strategy == 'slot' and savestate and savestate.loadslot then return pcall(savestate.loadslot, idx)
  elseif SAVE.strategy == 'file' and savestate and savestate.load then local path=string.format('%s/emerald_ai_slot_%d.State', CONFIG.save_dir, idx); return pcall(savestate.load, path) end
  return false
end
local function maybe_save_progress()
  if not SAVE.enabled then return end
  if PROGRESS.unique_cells_total - SAVE.last_saved_unique_cells >= CONFIG.savestate_save_every_new_cells then
    local idx=SAVE.next_slot; local ok=save_slot(idx)
    if ok then SAVE.saved_slots=math.max(SAVE.saved_slots, idx); SAVE.next_slot=(idx % CONFIG.savestate_slots)+1; SAVE.last_saved_unique_cells=PROGRESS.unique_cells_total; log_line(string.format('Savestate saved slot %d (unique=%d)', idx, PROGRESS.unique_cells_total)); journal_event('Savestate saved', string.format('slot=%d unique_cells=%d', idx, PROGRESS.unique_cells_total)) end
  end
end
local function maybe_reset_no_progress()
  if not SAVE.enabled then return end
  local frame=now_frame(); if (frame-PROGRESS.last_novelty_frame)>=CONFIG.no_novelty_reset_frames and SAVE.saved_slots>0 then local idx=math.random(1, math.min(SAVE.saved_slots, CONFIG.savestate_slots)); local ok=load_slot(idx); if ok then PROGRESS.last_novelty_frame=now_frame(); journal_event('Reset due to no progress', string.format('loaded slot %d', idx)) end end
end
local function detect_savestate_capabilities()
  if not CONFIG.savestate_enable or not savestate then return end
  if savestate.saveslot and savestate.loadslot then SAVE.strategy='slot'; SAVE.enabled=true elseif savestate.save and savestate.load then SAVE.strategy='file'; SAVE.enabled=true else SAVE.enabled=false end
end

-------------------------------------------------------------------------------
-- Actions & Environment Step
-------------------------------------------------------------------------------
local ACTIONS = { 'Up', 'Down', 'Left', 'Right', 'A', 'B' }
local function choose_action(state_hash, available_actions, epsilon)
  if math.random() < epsilon then return available_actions[math.random(1, #available_actions)] end
  local best_a, best_q = available_actions[1], -1e9
  for _, a in ipairs(available_actions) do local e=QTABLE[q_key(state_hash,a)]; local q=(e and e.q) or 0.0; if q>best_q then best_q=q; best_a=a end end
  return best_a
end
local function update_q(state_hash, action, reward, next_state_hash)
  local key=q_key(state_hash,action); local e=QTABLE[key] or {q=0.0,n=0}; local max_next=-1e9
  for _,def in ipairs(ACTION_DEFS) do local a=def.name; local en=QTABLE[q_key(next_state_hash,a)]; local q=(en and en.q) or 0.0; if q>max_next then max_next=q end end
  if max_next==-1e9 then max_next=0.0 end
  local lr,g=CONFIG.learning_rate, CONFIG.discount_factor; e.q = e.q + lr*(reward + g*max_next - e.q); e.n = e.n + 1; QTABLE[key]=e
end
local function step_action(action)
  local x0,y0=get_player_xy(); exec_action_by_name(action); local x1,y1=get_player_xy(); local reward=0.0
  if x0 and y0 and x1 and y1 then local moved=(x0~=x1) or (y0~=y1); if moved then reward=reward+CONFIG.small_move_reward end; local manhattan=(math.abs((x1 or 0)-(x0 or 0)) + math.abs((y1 or 0)-(y0 or 0))); if manhattan>=CONFIG.warp_distance_threshold then reward=reward+CONFIG.warp_reward; PROGRESS.last_warp_frame=now_frame(); journal_event('Warp/transition', string.format('delta=%d', manhattan)); take_screenshot('warp') end end
  reward = reward + novelty_reward_for_cell(coarse_cell(x1, y1))
  return { x=x1, y=y1, reward=reward }
end

-------------------------------------------------------------------------------
-- Epsilon schedule
-------------------------------------------------------------------------------
local function epsilon_for_frame(frame) local e0,emin,decay=CONFIG.epsilon_start,CONFIG.epsilon_min,CONFIG.epsilon_decay_frames; if frame>=decay then return emin end; local t=frame/decay; return e0 + (emin - e0)*t end

-------------------------------------------------------------------------------
-- Episodes & Journal
-------------------------------------------------------------------------------
local EP = { started_frame = 0, catches = 0, badges_mask = 0 }
local function start_episode() META.episode=(META.episode or 0)+1; META.total_episodes=(META.total_episodes or 0)+1; EP={ started_frame=now_frame(), catches=0, badges_mask=0 }; journal_append(string.format('\n\n## Episode %d\n', META.episode)); journal_append(string.format('- Start time: %s (F%d)\n', frames_to_hms(now_frame()), now_frame())); journal_append(string.format('- Epsilon start: %.3f, min: %.3f\n', CONFIG.epsilon_start, CONFIG.epsilon_min)); if SAVE.enabled then journal_append('- Savestates: enabled\n') else journal_append('- Savestates: disabled\n') end end
local function soft_reset_or_reload_start() local ok=false; if emu and emu.softreset then ok=pcall(emu.softreset) end; if not ok and client and client.reboot_core then ok=pcall(client.reboot_core) end; if not ok and SAVE.enabled and SAVE.saved_slots>0 then ok=load_slot(1) end; if ok then log_line('Episode reset executed') else log_line('Episode reset: fallback no-op') end end
local function end_episode(reason) local elapsed=now_frame()-(EP.started_frame or now_frame()); journal_append(string.format('- Episode end: %s (F%d)\n', frames_to_hms(now_frame()), now_frame())); journal_append(string.format('- Reason: %s\n', reason or 'completed')); journal_append(string.format('- Unique cells: %d\n', PROGRESS.unique_cells_total)); journal_append(string.format('- Catches: %d\n', EP.catches or 0)); journal_append(string.format('- Badges mask: 0x%04X\n', EP.badges_mask or 0)); journal_append(string.format('- Duration: %s\n', frames_to_hms(elapsed))); take_screenshot('episode_end'); save_qtable(); save_meta(); save_calib(); CONFIG.epsilon_start=math.max(CONFIG.epsilon_min, CONFIG.epsilon_start*CONFIG.lower_epsilon_per_episode); soft_reset_or_reload_start(); start_episode() end
local function check_episode_completion() if DETECT.badges.best_addr then local cur=read_u16_le(DETECT.badges.best_addr); if cur then local mask=CONFIG.episode_badge_mask; if bit_and(cur,mask)==mask then end_episode('All badges earned'); return true end end end return false end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local RECENT = { catch_attempt_frame = nil, catch_snap_before = nil, last_xy = { x=nil, y=nil, stagnant=0 }, last_ball_throw_frame = nil }

local function init()
  math.randomseed(os.time() % 2147483647)
  load_meta(); load_qtable(); load_calib(); try_load_hints(); detect_savestate_capabilities()
  if SAVE.enabled then save_slot(1); SAVE.saved_slots=1; SAVE.next_slot=2 end
  if CALIB.caught_counter_addr then DETECT.catch.best_addr=CALIB.caught_counter_addr; DETECT.catch.prev_value=read_u16_le(CALIB.caught_counter_addr) end
  if CALIB.map_id_addr then DETECT.map.best_addr=CALIB.map_id_addr; DETECT.map.prev_value=read_u16_le(CALIB.map_id_addr) end
  if CALIB.badge_flags_addr then DETECT.badges.best_addr=CALIB.badge_flags_addr; DETECT.badges.prev_value=read_u16_le(CALIB.badge_flags_addr) end
  if DETECT.bag.ball_count_addr then DETECT.bag.prev_count = read_u16_le(DETECT.bag.ball_count_addr) end
  log_line('Init done; emulator='..EMU.name)
  start_episode()
end

local function main_loop()
  local last_save_frame=now_frame(); local steps=0
  while true do
    local frame=now_frame(); if (frame-last_save_frame)>=CONFIG.autosave_interval_frames then save_qtable(); save_meta(); save_calib(); last_save_frame=frame; log_line('Autosaved state') end

    local x,y=get_player_xy()
    if not x or not y then if frame % 2000 == 0 then auto_calibrate_xy() end; exec_action_by_name(ACTIONS[math.random(1,#ACTIONS)])
    else
      update_phase(x,y)
      local available_actions = get_available_actions_for_phase(PHASE.mode)
      local state_hash = hash_state(x,y)
      local eps = epsilon_for_frame(frame)

      -- Pre-intent logs for macros
      local id, inCenter, inMart = get_map_context()
      local willHeal = inCenter and needs_heal() and (frame % CONFIG.heal_check_interval_frames == 0)
      local willBuy = inMart and balls_low()

      local action = choose_action(state_hash, available_actions, eps)

      -- Ball cooldown gating
      if action == 'BagThrowBallQuick' and RECENT.last_ball_throw_frame and (frame-RECENT.last_ball_throw_frame) < CONFIG.ball_throw_cooldown_frames then
        local alt={}; for _,a in ipairs(available_actions) do if a ~= 'BagThrowBallQuick' then table.insert(alt,a) end end; action=(#alt>0) and alt[math.random(1,#alt)] or 'A'
      end
      -- Prefer intent macros if conditions
      if willHeal then action = 'CenterHeal' end
      if willBuy and (action ~= 'CenterHeal') then action = 'MartBuyBalls10' end

      if action == 'BagThrowBallQuick' then
        local snaps={}; for _,region in ipairs(CONFIG.calib_scan_regions) do snaps[region.name]=snapshot_region(region) end
        RECENT.catch_attempt_frame=frame; RECENT.catch_snap_before=snaps; RECENT.last_ball_throw_frame=frame
        journal_event('Ball throw', 'attempt'); take_screenshot('ball')
      elseif action == 'CenterHeal' then
        journal_event('Center heal macro', 'executed (low party HP)'); take_screenshot('center')
      elseif action == 'MartBuyBalls10' or action == 'MartBuyBallsQuick' then
        journal_event('Mart buy balls macro', 'executed'); take_screenshot('mart')
      elseif action == 'BattleSwitchNext' then
        journal_event('Battle switch', 'attempt next');
      end

      local outcome = step_action(action)

      -- Phase shaping & catch scan
      local before_mode=PHASE.mode; local _, new_mode = update_phase(outcome.x, outcome.y)
      if before_mode=='interaction' and new_mode=='overworld' then
        outcome.reward = outcome.reward + CONFIG.interaction_exit_reward
        if RECENT.catch_attempt_frame and (frame-RECENT.catch_attempt_frame)<=CONFIG.catch_attempt_window_frames and RECENT.catch_snap_before then
          local after_snaps={}; for _,region in ipairs(CONFIG.calib_scan_regions) do after_snaps[region.name]=snapshot_region(region) end
          on_possible_catch_event(RECENT.catch_snap_before, after_snaps, CONFIG.calib_scan_regions)
          RECENT.catch_attempt_frame=nil; RECENT.catch_snap_before=nil
        end
      elseif new_mode=='interaction' and (PHASE.frames_in_mode % 60 == 0) then outcome.reward = outcome.reward + CONFIG.long_interaction_penalty end

      -- Catch reward via counter increment
      if DETECT.catch.best_addr then
        local cur=read_u16_le(DETECT.catch.best_addr)
        if cur and DETECT.catch.prev_value and cur>DETECT.catch.prev_value then outcome.reward=outcome.reward+CONFIG.catch_reward; EP.catches=(EP.catches or 0)+1; journal_event('Catch', string.format('caught_counter=%d', cur)); take_screenshot('caught') end
        DETECT.catch.prev_value = cur or DETECT.catch.prev_value
      end

      -- Map change reward / journal / Frontier
      if CALIB.map_id_addr then
        local oldm=DETECT.map.prev_value; local curm=read_u16_le(CALIB.map_id_addr)
        if curm and oldm and curm~=oldm then
          outcome.reward=outcome.reward+CONFIG.new_map_reward
          journal_event('Map change', string.format('%s(%d) -> %s(%d)', map_name(oldm), oldm, map_name(curm), curm)); take_screenshot('map')
          local prevF=((CONFIG.map_sets.frontier_map_ids or {})[oldm] == true); local curF=((CONFIG.map_sets.frontier_map_ids or {})[curm] == true)
          if (not prevF) and curF then outcome.reward=outcome.reward+(CONFIG.frontier_entry_reward or 0); journal_event('Entered Frontier', string.format('%s(%d)', map_name(curm), curm)); take_screenshot('frontier') end
        end
        DETECT.map.prev_value = curm or DETECT.map.prev_value
      end

      -- Badge gain reward / journal
      if CALIB.badge_flags_addr then
        local curb=read_u16_le(CALIB.badge_flags_addr)
        if curb and DETECT.badges.prev_value then local gained=bit_and(curb, bit_not16(DETECT.badges.prev_value)); if gained~=0 then outcome.reward=outcome.reward+CONFIG.badge_reward; EP.badges_mask=curb; journal_event('Badge gained', string.format('flags=0x%04X gained=0x%04X', curb, gained)); take_screenshot('badge') end end
        DETECT.badges.prev_value = curb or DETECT.badges.prev_value
      end

      -- Bag count journaling
      if DETECT.bag.ball_count_addr then
        local curBalls = read_u16_le(DETECT.bag.ball_count_addr)
        if curBalls and DETECT.bag.prev_count and curBalls > DETECT.bag.prev_count then journal_event('Balls purchased', string.format('+%d (now %d)', curBalls - DETECT.bag.prev_count, curBalls)) end
        DETECT.bag.prev_count = curBalls or DETECT.bag.prev_count
      end

      local next_state_hash = hash_state(outcome.x, outcome.y)
      update_q(state_hash, action, outcome.reward, next_state_hash)

      META.total_steps=(META.total_steps or 0)+1; steps=steps+1

      -- Stuck detection
      if RECENT.last_xy.x and RECENT.last_xy.y and outcome.x==RECENT.last_xy.x and outcome.y==RECENT.last_xy.y then RECENT.last_xy.stagnant=RECENT.last_xy.stagnant+1 else RECENT.last_xy.stagnant=0 end
      RECENT.last_xy.x, RECENT.last_xy.y = outcome.x, outcome.y
      if RECENT.last_xy.stagnant > 120 then local s=hash_state(outcome.x,outcome.y); update_q(s, action, CONFIG.stuck_penalty, s); exec_action_by_name('A'); exec_action_by_name(ACTIONS[math.random(1,4)]); RECENT.last_xy.stagnant=0 end

      -- Episode completion
      if check_episode_completion() then end
    end

    -- Savestates
    maybe_save_progress(); maybe_reset_no_progress()

    if steps % CONFIG.log_every_n_steps == 0 then log_line(string.format('step=%d eps=%.3f phase=%s(%d) unique=%d', steps, epsilon_for_frame(frame), PHASE.mode, PHASE.frames_in_mode, PROGRESS.unique_cells_total)) end

    emu.frameadvance()
  end
end

-------------------------------------------------------------------------------
-- Entry
-------------------------------------------------------------------------------
init()
main_loop()
