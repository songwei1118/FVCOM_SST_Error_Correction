clear; clc;
output_dir = './output_2010_2023_full_year_strict';
if ~exist(output_dir,'dir'), mkdir(output_dir); end

lon_min = 117; lon_max = 127;
lat_min = 35.5;  lat_max = 41.5;

OSTIA_all_years = [];
all_dates = [];

for yyyy = 2010:2023
   fn = sprintf('OSTIA_%d0101_%d0101_nrt.nc', yyyy, yyyy+1);
   fp = fullfile('Z:\OSTIA\OSTIA', fn);

   if ~isfile(fp)
       warning('Missing NRT OSTIA file for year %d: %s, skip', yyyy, fn);
       continue;
   end

fprintf('Processing OSTIA year %d (NRT only)...\n', yyyy);

   lon_var_name = 'longitude';
   lat_var_name = 'latitude';

    try
        lon_sat = double(ncread(fp, lon_var_name));
        lat_sat = double(ncread(fp, lat_var_name));
        sst = double(ncread(fp,'analysed_sst'));
        fv = ncreadatt(fp,'analysed_sst','_FillValue');
        sst(sst==fv) = NaN;

        lon_idx = lon_sat >= lon_min & lon_sat <= lon_max;
        lat_idx = lat_sat >= lat_min & lat_sat <= lat_max;
        
        if ~any(lon_idx) || ~any(lat_idx)
            warning('No target-region data in OSTIA for year %d, skip', yyyy);
            continue;
        end
        
        lon_t = lon_sat(lon_idx);
        lat_t = lat_sat(lat_idx);

        dates = datetime(yyyy,1,1) + days(0:size(sst,3)-1);
        valid = true(size(dates));
        
        if ~any(valid)
            warning('No valid OSTIA data for year %d, skip', yyyy);
            continue;
        end
        
        tmp = sst(lon_idx,lat_idx,valid) - 273.15;
        OSTIA_all_years = cat(3, OSTIA_all_years, tmp);
        all_dates = [all_dates; dates(valid)'];
        
       fprintf('  OK: NRT read, data size: %dx%dx%d\n', size(tmp,1), size(tmp,2), size(tmp,3));

        
    catch ME
        warning('Failed to read OSTIA for year %d: %s', yyyy, ME.message);
        
        fprintf('  File path: %s\n', fp);
        fprintf('  Tried lon/lat variables: %s, %s\n', lon_var_name, lat_var_name);
        
        try
            info = ncinfo(fp);
            vars = {info.Variables.Name};
            fprintf('  Variables in file: %s\n', strjoin(vars, ', '));
        catch
            fprintf('  Unable to get file variable info\n');
        end
        continue;
    end
end

if isempty(OSTIA_all_years)
    error('Failed to read any OSTIA data. Check file paths and formats');
else
    fprintf('OK: read OSTIA for %d years, total size: %dx%dx%d\n', ...
        length(unique(year(all_dates))), size(OSTIA_all_years,1), size(OSTIA_all_years,2), size(OSTIA_all_years,3));
end

fprintf('\n=== Strict time-range filter ===\n');
date_mask = year(all_dates) <= 2023;
original_days = length(all_dates);
all_dates = all_dates(date_mask);
OSTIA_all_years = OSTIA_all_years(:,:,date_mask);

fprintf('OK: time range filtered from %d days to %d days (removed %d days in 2024)\n', ...
    original_days, length(all_dates), original_days - length(all_dates));
fprintf('OK: strict time range: %s to %s\n', ...
    datestr(all_dates(1)), datestr(all_dates(end)));

save(fullfile(output_dir,'OSTIA_all_years_2010_2023_full_year.mat'), 'OSTIA_all_years','all_dates','lon_t','lat_t');

total_days = size(OSTIA_all_years,3);
valid_count = sum(~isnan(OSTIA_all_years), 3);
sea_mask = double(valid_count > total_days * 0.9);
sea_mask(sea_mask == 0) = NaN;
save(fullfile(output_dir,'sea_mask_2010_2023_full_year.mat'),'sea_mask');

[LON_t, LAT_t] = ndgrid(lon_t, lat_t);

wind_vars = {'U10','V10'};
heat_vars = {'short_wave','long_wave','air_pressure','air_temperature'};
all_vars = [wind_vars, heat_vars];

target_data = struct();
for v = all_vars, target_data.(v{1}) = []; end

fprintf('=== Time alignment: use OSTIA dates as the common time axis ===\n');
full_year_dates = all_dates;
total_days = length(full_year_dates);
fprintf('OK: common time axis: %s to %s, %d days\n', ...
    datestr(full_year_dates(1)), datestr(full_year_dates(end)), total_days);

for v = all_vars
    target_data.(v{1}) = NaN(length(lon_t), length(lat_t), total_days);
end

for yyyy = 2009:2023
    if yyyy < 2010 || yyyy > 2023
        continue;
    end
    
    wf = sprintf('Z:/data_input/era5_forcing/ecs3_wind_era5_%d.nc', yyyy);
    hf = sprintf('Z:/data_input/era5_forcing/ecs3_heat_cal_era5_%d.nc', yyyy);
    if ~isfile(wf)||~isfile(hf), warning('Skip year %d', yyyy); continue; end

    fprintf('Processing ERA5 %d\n', yyyy);

    tnum = double(ncread(wf,'time')) + datenum(1858,11,17);
    fd = datetime(tnum,'ConvertFrom','datenum');

    xlon = double(ncread(wf,'XLONG'));
    xlat = double(ncread(wf,'XLAT'));
    U10 = double(ncread(wf,'U10'));
    V10 = double(ncread(wf,'V10'));
    SW  = double(ncread(hf,'short_wave'));
    LW  = double(ncread(hf,'long_wave'));
    AP  = double(ncread(hf,'air_pressure'));
    AT  = double(ncread(hf,'air_temperature'));

    for d = 1:total_days
        cur_date = full_year_dates(d);
        if yyyy ~= year(cur_date)
            continue;
        end
        idxs = find(dateshift(fd, 'start', 'day') == cur_date);
        if isempty(idxs), continue; end

        for v = all_vars
            switch v{1}
                case 'U10', raw = U10;
                case 'V10', raw = V10;
                case 'short_wave', raw = SW;
                case 'long_wave',  raw = LW;
                case 'air_pressure', raw = AP;
                case 'air_temperature', raw = AT;
            end
            slice = mean(raw(:,:,idxs),3,'omitnan');
            x = xlon(:); y = xlat(:); z = slice(:);
            valid = ~isnan(z);
            if sum(valid) < 3, continue; end
            F = scatteredInterpolant(x(valid), y(valid), z(valid), 'linear','nearest');
            interp = F(LON_t, LAT_t);
            target_data.(v{1})(:,:,d) = interp;
        end
    end
end
ERA5 = target_data;
save(fullfile(output_dir,'era5_2010_2023_full_year.mat'),'ERA5','-v7.3');
save(fullfile(output_dir, 'full_year_dates_2010_2023_full_year.mat'), 'full_year_dates');

fprintf('\n========== Processing FVCOM (full year, 2010-2023) ==========\n');
FVCOM = struct();
for v = {'temp','km','u','v'}
    FVCOM.(v{1}) = NaN([length(lon_t), length(lat_t), total_days], 'single');
end

actual_days_mask = false(total_days,1);

for yyyy = 2010:2023
    fvcom_root = 'Z:/SDP/output_river1/';
    year_folder = fullfile(fvcom_root, num2str(yyyy));
    file_pattern = 'sdp_avg_*.nc';
    
    if ~exist(year_folder, 'dir')
        warning('FVCOM folder not found for year %d: %s', yyyy, year_folder);
        continue;
    end
    
    files = dir(fullfile(year_folder, file_pattern));
    if isempty(files)
        warning('No %s files found in %s', file_pattern, year_folder);
        continue;
    end
    
    fprintf('Processing FVCOM year %d, found %d files in %s\n', yyyy, length(files), year_folder);
    
    for f = 1:length(files)
        name = files(f).name;
        path = fullfile(files(f).folder, name);

        try
            tnum = double(ncread(path, 'time')); 
            file_date = datetime(tnum + datenum(1858,11,17), 'ConvertFrom', 'datenum');

            file_date = dateshift(file_date(1), 'start', 'day');

            day_idx = find(full_year_dates == file_date);
            if isempty(day_idx)
                continue;
            end

            temp = double(ncread(path,'temp'));
            km   = double(ncread(path,'km'));
            u    = double(ncread(path,'u'));
            v    = double(ncread(path,'v'));
            lon  = double(ncread(path,'lon'));
            lat  = double(ncread(path,'lat'));
            lonc = double(ncread(path,'lonc'));
            latc = double(ncread(path,'latc'));

            for varname = {'temp','km','u','v'}
                try
                    fv = ncreadatt(path, varname{1}, '_FillValue');
                catch
                    fv = [];
                end
                if ~isempty(fv)
                    eval(sprintf('%s(%s==fv) = NaN;', varname{1}, varname{1}));
                end
            end

            temp = squeeze(temp(:,1,1));
            km   = squeeze(km(:,1,1));
            u    = squeeze(u(:,1,1));
            v    = squeeze(v(:,1,1));

            if sum(~isnan(temp)) > 5
                F = scatteredInterpolant(lon, lat, temp, 'linear', 'nearest');
                FVCOM.temp(:,:,day_idx) = single(F(LON_t, LAT_t));
            end
            if sum(~isnan(km)) > 5
                F = scatteredInterpolant(lon, lat, km, 'linear', 'nearest');
                FVCOM.km(:,:,day_idx) = single(F(LON_t, LAT_t));
            end
            if sum(~isnan(u)) > 5
                F = scatteredInterpolant(lonc, latc, u, 'linear', 'nearest');
                FVCOM.u(:,:,day_idx) = single(F(LON_t, LAT_t));
            end
            if sum(~isnan(v)) > 5
                F = scatteredInterpolant(lonc, latc, v, 'linear', 'nearest');
                FVCOM.v(:,:,day_idx) = single(F(LON_t, LAT_t));
            end

            actual_days_mask(day_idx) = true;

            if mod(day_idx,50)==0 || day_idx==1
                fprintf('OK %s (%s) interpolated -> day_idx=%d\n', name, datestr(file_date), day_idx);
            end

        catch ME
            fprintf('Skip %s: %s\n', name, ME.message);
        end
    end
end

save(fullfile(output_dir,'FVCOM_2010_2023_full_year.mat'),'FVCOM','-v7.3');
save(fullfile(output_dir,'fvcom_day_mask_2010_2023_full_year.mat'),'actual_days_mask','full_year_dates');

fprintf('\n========== Merge all data (with time check) ==========\n');

fprintf('=== Time-range consistency check ===\n');

if length(all_dates) ~= length(full_year_dates)
    error('Fatal: OSTIA date length (%d) does not match full_year_dates length (%d)', ...
        length(all_dates), length(full_year_dates));
end

date_mismatch_mask = (all_dates ~= full_year_dates);
date_mismatch_count = sum(date_mismatch_mask);

if date_mismatch_count > 0
    fprintf('Warning: %d days have mismatched dates\n', date_mismatch_count);
    
    mismatch_indices = find(date_mismatch_mask, 5);
    for i = 1:min(5, length(mismatch_indices))
        idx = mismatch_indices(i);
        fprintf('  Day %d: OSTIA=%s, full_year_dates=%s\n', ...
            idx, datestr(all_dates(idx)), datestr(full_year_dates(idx)));
    end
    
    response = input('Continue processing? (y/n): ', 's');
    if ~strcmpi(response, 'y')
        error('Stopped by user');
    end
else
    fprintf('OK: OSTIA dates match full_year_dates\n');
end

fprintf('=== Year-range check ===\n');

ostia_years = unique(year(full_year_dates));
if min(ostia_years) == 2010 && max(ostia_years) == 2023
    fprintf('OK: year range is 2010-2023\n');
else
    error('Incorrect year range: %d-%d', min(ostia_years), max(ostia_years));
end

expected_days = 11 * 365 + 3 * 366;
if total_days == expected_days
    fprintf('OK: total days = %d\n', total_days);
else
    warning('Total days mismatch: expected %d, got %d', expected_days, total_days);
end

valid_idx = find(actual_days_mask);
valid_days = length(valid_idx);

fprintf('\n=== Time alignment check (first sources) ===\n');
for i = 1:min(5,valid_days)
    t = valid_idx(i);
    fprintf('Day %d: OSTIA=%s | ERA5=%s | FVCOM=%s\n',...
    i, datestr(all_dates(t)), datestr(full_year_dates(t)), datestr(full_year_dates(t)));
end

[lon_dim, lat_dim] = size(OSTIA_all_years(:,:,1));
data = zeros(valid_days,14,lon_dim,lat_dim,'single');
[LatGrid, LonGrid] = meshgrid(lat_t, lon_t);

for n = 1:valid_days
    t = valid_idx(n);
    slab = zeros(14,lon_dim,lat_dim,'single');
    try
        slab(1,:,:)  = OSTIA_all_years(:,:,t);
        slab(2,:,:)  = ERA5.U10(:,:,t);
        slab(3,:,:)  = ERA5.V10(:,:,t);
        slab(4,:,:)  = ERA5.short_wave(:,:,t);
        slab(5,:,:)  = ERA5.long_wave(:,:,t);
        slab(6,:,:)  = ERA5.air_pressure(:,:,t);
        slab(7,:,:)  = FVCOM.temp(:,:,t);
        slab(8,:,:)  = FVCOM.km(:,:,t);
        slab(9,:,:)  = FVCOM.u(:,:,t);
        slab(10,:,:) = FVCOM.v(:,:,t);
        slab(11,:,:) = LonGrid;
        slab(12,:,:) = LatGrid;
        slab(13,:,:) = (n-1)/(valid_days-1);
        slab(14,:,:) = ERA5.air_temperature(:,:,t);

        for v = [1:10, 14]
            sl = squeeze(slab(v,:,:));
            sl = sl .* sea_mask;
            slab(v,:,:) = sl;
        end

        data(n,:,:,:) = slab;
    catch ME
        fprintf('Merge failed for day %d (raw t=%d): %s\n', n, t, ME.message);
    end
end

var_names = {'sat_T','U10','V10','short_wave','long_wave','air_pressure',...
             'FVCOM_T','FVCOM_km','FVCOM_u','FVCOM_v','lon','lat','time','air_temperature'};
var_info = struct('names',{var_names},...
                 'dimensions',{[valid_days,14,lon_dim,lat_dim]},...
                 'dates',{full_year_dates(valid_idx)});
save(fullfile(output_dir,'data_with_mask_2010_2023_full_year.mat'),'data','var_info','valid_idx','full_year_dates','-v7.3');

output_fig_dir = fullfile(output_dir, 'figures_2010_2023_full_year');
if ~exist(output_fig_dir, 'dir'), mkdir(output_fig_dir); end

test_day = 1;
figure('Position',[100,100,1200,400]);
subplot(1,3,1);
imagesc(lon_t, lat_t, squeeze(OSTIA_all_years(:,:,valid_idx(test_day)))');
title(sprintf('OSTIA SST - %s', datestr(full_year_dates(valid_idx(test_day)))));
colorbar; axis xy;

subplot(1,3,2);
imagesc(lon_t, lat_t, squeeze(ERA5.U10(:,:,valid_idx(test_day)))');
title(sprintf('ERA5 U10 - %s', datestr(full_year_dates(valid_idx(test_day)))));
colorbar; axis xy;

subplot(1,3,3);
imagesc(lon_t, lat_t, squeeze(FVCOM.temp(:,:,valid_idx(test_day)))');
title(sprintf('FVCOM Temp - %s', datestr(full_year_dates(valid_idx(test_day)))));
colorbar; axis xy;

saveas(gcf, fullfile(output_fig_dir, 'time_alignment_validation_2010_2023_full_year.png'));
close;

fprintf('OK: all processing finished. Results saved in %s\n', output_dir);
fprintf('OK: strict time range 2010-2023, %d days\n', total_days);

fprintf('\n========== Three-source time alignment check ==========\n');

full_year_dates_file = fullfile(output_dir, 'full_year_dates_2010_2023_full_year.mat');
fvcom_mask_file   = fullfile(output_dir, 'fvcom_day_mask_2010_2023_full_year.mat');

if ~exist(full_year_dates_file, 'file')
    error('Cannot find full_year_dates_2010_2023_full_year.mat. Generate the date file first');
end
if ~exist(fvcom_mask_file, 'file')
    error('Cannot find fvcom_day_mask_2010_2023_full_year.mat. Generate the FVCOM mask file first');
end

load(full_year_dates_file, 'full_year_dates');
load(fvcom_mask_file, 'actual_days_mask');

years_range = unique(year(full_year_dates));
fprintf('=== Final year-range check ===\n');
fprintf('OK: year range: %d-%d\n', min(years_range), max(years_range));
fprintf('OK: total days: %d\n', length(full_year_dates));

if min(years_range) == 2010 && max(years_range) == 2023
    fprintf('OK: strict time range check passed: 2010-2023\n');
else
    warning('Time range check failed');
end

rng(0);
sample_days = sort(randperm(length(full_year_dates), 10));

fprintf('%-6s | %-12s | %-8s | %-8s | %-8s\n', ...
    'Index', 'full_year_dates', 'OSTIA', 'ERA5', 'FVCOM');
fprintf(repmat('-', 1, 50)); fprintf('\n');

for i = 1:length(sample_days)
    idx = sample_days(i);
    date_str = datestr(full_year_dates(idx), 'yyyy-mm-dd');

    ostia_flag = ~all(isnan(OSTIA_all_years(:,:,idx)), 'all');

    era5_flag = ~all(isnan(ERA5.air_temperature(:,:,idx)), 'all');

    fvcom_flag = actual_days_mask(idx);

    fprintf('%-6d | %-12s | %-8s | %-8s | %-8s\n', ...
        idx, date_str, ...
        logical2str(ostia_flag), ...
        logical2str(era5_flag), ...
        logical2str(fvcom_flag));
end

ostia_missing = arrayfun(@(d) all(isnan(OSTIA_all_years(:,:,d)), 'all'), ...
                         1:length(full_year_dates))';
era5_missing  = arrayfun(@(d) all(isnan(ERA5.air_temperature(:,:,d)), 'all'), ...
                         1:length(full_year_dates))';

mismatch_mask = (actual_days_mask ~= (~ostia_missing & ~era5_missing));

if any(mismatch_mask)
    fprintf('\nWarning: %d days are not fully consistent across the three sources\n', sum(mismatch_mask));
else
    fprintf('\nOK: alignment check passed: OSTIA, ERA5, and FVCOM dates match\n');
end

ostia_missing = arrayfun(@(d) all(isnan(OSTIA_all_years(:,:,d)), 'all'), 1:length(full_year_dates))';
era5_missing  = arrayfun(@(d) all(isnan(ERA5.air_temperature(:,:,d)), 'all'), 1:length(full_year_dates))';
fvcom_missing = ~actual_days_mask;

aligned_mask = ~ostia_missing & ~era5_missing & actual_days_mask;

total_days      = length(full_year_dates);
num_ostia_miss  = sum(ostia_missing);
num_era5_miss   = sum(era5_missing);
num_fvcom_miss  = sum(fvcom_missing);
num_aligned     = sum(aligned_mask);
num_not_aligned = total_days - num_aligned;

fprintf('\n===== Alignment statistics =====\n');
fprintf('Total days: %d\n', total_days);
fprintf('OSTIA missing days: %d\n', num_ostia_miss);
fprintf('ERA5 missing days: %d\n', num_era5_miss);
fprintf('FVCOM missing days: %d\n', num_fvcom_miss);
fprintf('Days aligned across three sources: %d\n', num_aligned);
fprintf('Days not aligned: %d\n', num_not_aligned);

T = table;
T.Date = full_year_dates;
T.OSTIA = ~ostia_missing;
T.ERA5  = ~era5_missing;
T.FVCOM = actual_days_mask;
T.Aligned = aligned_mask;

T.OSTIA = repmat({'Y present'}, height(T),1);
T.OSTIA(ostia_missing) = {'N missing'};
T.ERA5  = repmat({'Y present'}, height(T),1);
T.ERA5(era5_missing) = {'N missing'};
T.FVCOM = repmat({'Y present'}, height(T),1);
T.FVCOM(fvcom_missing) = {'N missing'};
T.Aligned = repmat({'Y aligned'}, height(T),1);
T.Aligned(~aligned_mask) = {'N not aligned'};

disp('=== Alignment for the first 10 days ===');
disp(T(1:10,:));

save_excel = true;
if save_excel
    output_file = fullfile(pwd,'data_alignment_check_full_year.xlsx');
    writetable(T, output_file);
    fprintf('OK: alignment table saved to: %s\n', output_file);
end

function str = logical2str(flag)
    if flag
        str = 'Y present';
    else
        str = 'N missing';
    end
end
