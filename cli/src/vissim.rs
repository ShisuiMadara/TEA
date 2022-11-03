use clap::{arg, Command};

pub (crate) fn cli() -> Command {

    Command::new("vissim")
        .about("VISSIM callibration software.")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .allow_external_subcommands(true)
        .subcommand(
            Command::new("start")
                .short_flag('s')
                .about("start callibration")
                .arg(arg!(<File> "Location of data to calibrate."))
                .arg_required_else_help(true),
        )
}