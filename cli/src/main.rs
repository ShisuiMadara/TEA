mod vissim;

fn main() {
    let matches = vissim::cli().get_matches();

    match matches.subcommand() {
        Some(("start", sub_matches)) => {
            println!(
                "Callibrating on {}",
                sub_matches.get_one::<String>("File").expect("required")
            );
        }
        _ => unreachable!(), 
    }

}
