SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS score_prediction_points;
DROP TABLE IF EXISTS analysis_steps;
DROP TABLE IF EXISTS analysis_details;
DROP TABLE IF EXISTS sleep_time_details;
DROP TABLE IF EXISTS analyzed_sound_events;
DROP TABLE IF EXISTS analyzed_sleep_levels;
DROP TABLE IF EXISTS daily_reports;
DROP TABLE IF EXISTS periodic_reports;
DROP TABLE IF EXISTS sleep_sessions;
DROP TABLE IF EXISTS users;
SET FOREIGN_KEY_CHECKS = 1;

create table users (
	user_no bigint primary key,
	name varchar(64),
	nickname varchar(64),
	email varchar(128) not null,
	password varchar(64) not null,
	gender enum('MALE', 'FEMALE')
);

create table sleep_sessions (
    sleep_session_no bigint auto_increment,
    user_no bigint not null,
    finished_at datetime default null,
    created_at datetime not null default current_timestamp,

    constraint pk_sleep_sessions primary key (sleep_session_no),
    constraint fk_sleep_sessions_user_no
        foreign key (user_no) references users(user_no)
);

create table daily_reports (
    sleep_session_no bigint,
    memo varchar(255),
    user_no bigint not null,
    created_at datetime not null default current_timestamp,

    constraint pk_daily_reports primary key (sleep_session_no),
    constraint fk_daily_reports_sleep_session_no
        foreign key (sleep_session_no) references sleep_sessions(sleep_session_no)
        on delete cascade,
    constraint fk_daily_reports_user_no
        foreign key (user_no) references users(user_no)
);

create table sleep_time_details (
    sleep_session_no bigint,

    deep_sleep_minutes smallint unsigned not null,
    light_sleep_minutes smallint unsigned not null,
    rem_sleep_minutes smallint unsigned not null,

    deep_sleep_ratio decimal(4, 1) not null,
    light_sleep_ratio decimal(4, 1) not null,
    rem_sleep_ratio decimal(4, 1) not null,

    constraint pk_sleep_time_details primary key (sleep_session_no),
    constraint fk_sleep_time_details_sleep_session_no
        foreign key (sleep_session_no) references daily_reports(sleep_session_no)
        on delete cascade
);

create table analysis_details (
    analysis_detail_no bigint auto_increment,
    sleep_session_no bigint not null,

    title varchar(255) not null,
    description varchar(255) not null,
    difficulty enum('EASY', 'MEDIUM', 'HARD') not null,
    effect enum('LOW', 'MEDIUM', 'HIGH') not null,

    constraint pk_analysis_details primary key (analysis_detail_no),
    constraint fk_analysis_details_sleep_session_no
        foreign key (sleep_session_no) references daily_reports(sleep_session_no)
        on delete cascade
);

create table analysis_steps (
    analysis_step_no bigint auto_increment,
    analysis_detail_no bigint not null,

    step_index smallint unsigned not null,
    content varchar(255) not null,

    constraint pk_analysis_steps primary key (analysis_step_no),
    constraint fk_analysis_steps_analysis_detail_no
        foreign key (analysis_detail_no) 
        references analysis_details(analysis_detail_no)
        on delete cascade
        
);

create table periodic_reports (
    periodic_report_no bigint auto_increment,
    user_no bigint not null,

    sleep_session_count smallint unsigned not null default 0,

    duration_type enum('WEEKLY', 'MONTHLY') not null,
    period_started_at date not null,

    total_score smallint unsigned not null default 0,
    total_sleep_time smallint unsigned not null default 0,

    total_bed_time_minutes smallint unsigned not null default 0,
    total_deep_sleep_time_minutes smallint unsigned not null default 0,
    total_light_sleep_time_minutes smallint unsigned not null default 0,
    total_rem_sleep_time_minutes smallint unsigned not null default 0,

    improvement varchar(500),
    weakness varchar(500),
    recommendation varchar(500),
    score_prediction_description varchar(500),

    constraint pk_periodic_report primary key (periodic_report_no),
    constraint fk_periodic_report_user_no
        foreign key (user_no) references users(user_no)
);

create table score_prediction_points (
    score_prediction_point_no bigint auto_increment,
    periodic_report_no bigint not null,
    date_index date not null,
    score smallint not null,

    constraint pk_score_prediction_points primary key (score_prediction_point_no),
    constraint fk_score_predictions_points_periodic_report_no
        foreign key (periodic_report_no)
        references periodic_reports(periodic_report_no)
        on delete cascade
);

create table analyzed_sound_events (
    analyzed_sound_event_no bigint, -- tsid
    sleep_session_no bigint,
    event enum('SNORE', 'BABY_CRYING', 'COUGH', 'MOUTH_BREATHING', 'ANIMAL_NOISE', 'CAR_HORN'),
    recorded_at datetime(6) not null,

    constraint pk_analyzed_sound_events primary key (analyzed_sound_event_no),
    constraint fk_analyzed_sound_events_sleep_session_no
        foreign key (sleep_session_no) references sleep_sessions(sleep_session_no)
);

create table analyzed_sleep_levels (
    analyzed_sleep_level_no bigint, -- tsid
    sleep_session_no bigint,
    level smallint unsigned,
    recorded_at datetime(6) not null,

    constraint pk_analyzed_sleep_levels primary key (analyzed_sleep_level_no),
    constraint fk_analyzed_sleep_levels_session
        foreign key (sleep_session_no) references sleep_sessions(sleep_session_no),
    constraint chk_analyzed_sleep_levels_level
		    check (level between 0 and 6)
);
